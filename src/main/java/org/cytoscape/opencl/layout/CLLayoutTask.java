package org.cytoscape.opencl.layout;

import java.util.List;
import java.util.Random;
import java.util.Set;

import org.cytoscape.model.CyNode;
import org.cytoscape.opencl.cycl.*;
import org.cytoscape.opencl.cycl.CyCLDevice.DeviceTypes;
import org.cytoscape.view.layout.AbstractParallelPartitionLayoutTask;
import org.cytoscape.view.layout.LayoutNode;
import org.cytoscape.view.layout.LayoutPartition;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.model.View;
import org.cytoscape.work.undo.UndoSupport;


/**
 * This class implements the Prefuse force-directed layout algorithm using OpenCL.
 * 
 * @see <a href="http://prefuse.org">Prefuse web site</a>
 */
public class CLLayoutTask extends AbstractParallelPartitionLayoutTask 
{		
	private volatile Object sync = new Object();
	
	private final CLLayoutContext context;
	
	private final CyCLDevice device;
	private final CyCLProgram program;

	/**
	 * Creates a new ForceDirectedLayout object.
	 */
	public CLLayoutTask(final String displayName, 
						 CyNetworkView networkView, 
						 Set<View<CyNode>> nodesToLayOut,
						 final CLLayoutContext context,
						 String attrName, 
						 UndoSupport undo) 
	{
		super(displayName, context.singlePartition, networkView, nodesToLayOut, attrName, undo);

		this.context = context;

		edgeWeighter = context.edgeWeighter;
		edgeWeighter.setWeightAttribute(layoutAttribute);
		
		try
		{
			device = CyCL.getDevices().get(0);
		}
		catch (Exception e)
		{
			System.out.println("No OpenCL devices found, cannot do layout.");
			throw new RuntimeException();
		}
		
		//System.out.println("Layout will use " + device.name + ".");
		
		String[] kernelNames = new String[] 
				{
					"Init",
					"CalcForcesGravity",
					"PrepareEdgeRepulsion",
					"CalcForcesEdgeRepulsion",
					"CalcForcesSpringDrag",
					"IntegrateRK0",
					"IntegrateRK1",
					"IntegrateRK2",
					"IntegrateRK3",
					"IntegrateEuler"
				};
		
		CyCLProgram tryProgram;
		try
		{
			tryProgram = device.addProgram("PrefuseLayout", getClass().getResource("/LayoutKernels.cl"), kernelNames, null, false);
		}
		catch (Exception exc)
		{
			System.out.println("Could not load and compile OpenCL program, cannot do layout.");
			throw new RuntimeException();
		}
		program = tryProgram;
	}
	
	@Override
	public void layoutPartition(LayoutPartition partition) 
	{
		Layouter layouter = new Layouter();
		layouter.doLayout(partition);
	}
	
	@Override
	public String toString() 
	{
		return CLLayout.ALGORITHM_DISPLAY_NAME;
	}	

	private class Layouter
	{	
		public static final int requiredPadding = 16;
		
		// Node data
		private CyCLBuffer bufferNodePosX;
		private CyCLBuffer bufferNodePosY;
		private CyCLBuffer bufferNodeMass;
		
		// Edge data for spring forces
		private CyCLBuffer bufferEdges;
		private CyCLBuffer bufferEdgeCoeffs;
		private CyCLBuffer bufferEdgeLengths;
		private CyCLBuffer bufferEdgeOffsets;
		private CyCLBuffer bufferEdgeCounts;
		
		// Edge data for repulsive edges
		private CyCLBuffer bufferEdgeUniqueSources;
		private CyCLBuffer bufferEdgeUniqueTargets;
		private CyCLBuffer bufferEdgeStartX;
		private CyCLBuffer bufferEdgeStartY;
		private CyCLBuffer bufferEdgeTangentX;
		private CyCLBuffer bufferEdgeTangentY;
		private CyCLBuffer bufferEdgeCurrentLength;
		private CyCLBuffer bufferEdgeMassStart;
		private CyCLBuffer bufferEdgeMassEnd;
		
		private CyCLBuffer bufferForce;
		private CyCLBuffer bufferVelocity;
	
		private CyCLBuffer bufferNodeK;
		private CyCLBuffer bufferNodeL;
		
		private boolean buffersInitialized = false;
	
		public Layouter()
		{
			
		}

		public void doLayout(LayoutPartition part) 
		{
			synchronized (sync)
			{
				long startTime = System.currentTimeMillis();
				
				// Init positions to random or their current values
				if (context.fromScratch)
				{
					Random rand = new Random(123);
					List<LayoutNode> nodeList = part.getNodeList();
					
					for (LayoutNode node : nodeList)
					{
						node.setX((rand.nextFloat() - 0.5f) * 2f);
						node.setY((rand.nextFloat() - 0.5f) * 2f);
					}
				}
				
				// Calculate our edge weights
				part.calculateEdgeWeights();
				
				SlimNetwork slim = new SlimNetwork(part, 
												   context.isDeterministic, 
												   (float)context.defaultNodeMass, 
												   (float)context.defaultSpringCoefficient, 
												   (float)context.defaultSpringLength, 
												   edgeWeighter, 
												   requiredPadding);
				
				initializeBuffers(slim);
				
				if (taskMonitor != null)
					taskMonitor.setStatusMessage("Moving partition " + part.getPartitionNumber());
				
				// Initialize velocity to 0
				initializeSimulation(slim);
				
				// Perform layout
				float timestep = 1000f;
				for (int i = 0; i < context.numIterations && !cancelled; i++) 
				{
					// Gradually decrease time step as simulation converges
					float decrease = (1f - (float)i / (float)context.numIterations);
					timestep *= decrease;
					float step = timestep + 50f;
	
					advanceSimulation(step, false, slim);
				}
		
				if (context.numIterationsEdgeRepulsive > 0)
				{
					initializeSimulation(slim);
					
					timestep = 10f;
					for (int i = 0; i < context.numIterationsEdgeRepulsive && !cancelled; i++) 
					{
						// Gradually decrease time step as simulation converges
						float decrease = (1f - (float)i / (float)context.numIterations);
						timestep *= decrease;
						
						advanceSimulation(0.25f, true, slim);
					}
				}
				
				// Get positions back from CL device
				getPositions(slim);
				
				long stopTime = System.currentTimeMillis();
				//System.out.println(stopTime - startTime);
				
				// Update positions
				part.resetNodes(); // reset the nodes so we get the new average location
				for (LayoutNode ln: part.getNodeList())
				{
					if (!ln.isLocked()) 
					{
						int id = slim.nodeToIndex.get(ln);
						ln.setX(slim.nodePosX[id]);
						ln.setY(slim.nodePosY[id]);
						part.moveNodeToLocation(ln);
					}
				}
				
				// Release all CLBuffers
				freeBuffers();
			}
		}
	
		/***
		 * Allocates memory on GPU and fills it with network data
		 * @param slim Network data
		 * @param nodePosX X component of initial node positions
		 * @param nodePosY Y component of initial node positions
		 */
		private void initializeBuffers(SlimNetwork slim)
		{		
			// Initialize CLBuffers to hold node and edge information, and copy initial data to them
			bufferNodePosX = device.createBuffer(slim.nodePosX);
			bufferNodePosY = device.createBuffer(slim.nodePosY);
			bufferNodeMass = device.createBuffer(slim.nodeMass);
			
			bufferEdges = device.createBuffer(slim.edges);
			bufferEdgeCoeffs = device.createBuffer(slim.edgeCoeffs);
			bufferEdgeLengths = device.createBuffer(slim.edgeLengths);
			bufferEdgeOffsets = device.createBuffer(slim.edgeOffsetsSparse);
			bufferEdgeCounts = device.createBuffer(slim.edgeCounts);
			
			if (context.numIterationsEdgeRepulsive > 0)
			{
				bufferEdgeUniqueSources = device.createBuffer(slim.edgeUniqueSources);
				bufferEdgeUniqueTargets = device.createBuffer(slim.edgeUniqueTargets);
				// Init all with edgeMass because it has the padded tail set to 0:
				bufferEdgeStartX = device.createBuffer(slim.edgeMassStart);
				bufferEdgeStartY = device.createBuffer(slim.edgeMassStart);
				bufferEdgeTangentX = device.createBuffer(slim.edgeMassStart);
				bufferEdgeTangentY = device.createBuffer(slim.edgeMassStart);
				bufferEdgeCurrentLength = device.createBuffer(slim.edgeMassStart);
				bufferEdgeMassStart = device.createBuffer(slim.edgeMassStart);
				bufferEdgeMassEnd = device.createBuffer(slim.edgeMassEnd);
			}
			
			bufferForce = device.createBuffer(float.class, slim.numNodesPadded * 2);
			bufferVelocity = device.createBuffer(float.class, slim.numNodes * 2);
	
			bufferNodeK = device.createBuffer(float.class, slim.numNodes * 8);
			bufferNodeL = device.createBuffer(float.class, slim.numNodes * 6);
			
			buffersInitialized = true;
		}
	
		/***
		 * Releases all memory allocated on GPU
		 */
		private void freeBuffers()
		{
			if (!buffersInitialized)
				return;
			
			bufferNodePosX.free();
			bufferNodePosY.free();
			bufferNodeMass.free();
			bufferEdges.free();
			bufferEdgeCoeffs.free();
			bufferEdgeLengths.free();
			bufferEdgeOffsets.free();
			bufferEdgeCounts.free();
			
			if (context.numIterationsEdgeRepulsive > 0)
			{
				bufferEdgeStartX.free();
				bufferEdgeStartY.free();
				bufferEdgeStartX.free();
				bufferEdgeStartY.free();
				bufferEdgeTangentX.free();
				bufferEdgeTangentY.free();
				bufferEdgeCurrentLength.free();
				bufferEdgeMassStart.free();
				bufferEdgeMassEnd.free();
			}
			
			bufferForce.free();
			bufferVelocity.free();
			bufferNodeK.free();
			bufferNodeL.free();
	
			buffersInitialized = false;
		}
		
		/***
		 * Initializes velocity to 0
		 * @param slim Network data
		 */
		private void initializeSimulation(SlimNetwork slim)
		{
			program.getKernel("Init").execute(new long[] { slim.numNodes }, null, bufferVelocity, slim.numNodes);
		}
		
		/***
		 * Copy node positions from GPU to pre-initialized arrays
		 * @param nodePosX X component of node positions
		 * @param nodePosY Y component of node positions
		 */
		private void getPositions(SlimNetwork slim)
		{
			bufferNodePosX.getFromDevice(slim.nodePosX);
			bufferNodePosY.getFromDevice(slim.nodePosY);
		}
		
		/**
		 * Advances the simulation state by the given amount of time 
		 * using a Runge-Kutta 4th order integration scheme. All data are 
		 * stored in the CLBuffer objects initialized and populated previously.
		 * @param timestep Amount of virtual time to be simulated in this step.
		 */
		private void advanceSimulation(float timestep, boolean doEdgeRepulsion, SlimNetwork slim)
		{
			// Parallelization scheme is different for CPU and GPU kernel versions
			long[] dimsLocal = new long[]{ device.bestBlockSize };
			long[] dimsGlobal = new long[]{ nextMultipleOf(slim.numNodes, dimsLocal[0]) };
			
			calculateForces(doEdgeRepulsion, slim);
			
			program.getKernel("IntegrateRK0").execute(dimsGlobal, dimsLocal, 
						bufferNodePosX, bufferNodePosY, 
					   	bufferNodeMass, 
					   	bufferNodeK,
					   	bufferNodeL,
					   	bufferVelocity,
					    bufferForce,
					    timestep, 
					    slim.numNodes);

			calculateForces(doEdgeRepulsion, slim);
			
			program.getKernel("IntegrateRK1").execute(dimsGlobal, dimsLocal,
						bufferNodePosX, bufferNodePosY, 
					   	bufferNodeMass,
					   	bufferNodeK,
					   	bufferNodeL,
					   	bufferVelocity,
					    bufferForce,
					    1.0f,
					    timestep, 
					    slim.numNodes);

			calculateForces(doEdgeRepulsion, slim);
			
			program.getKernel("IntegrateRK2").execute(dimsGlobal, dimsLocal,
						bufferNodePosX, bufferNodePosY, 
					   	bufferNodeMass,
					   	bufferNodeK,
					   	bufferNodeL,
					   	bufferVelocity,
					    bufferForce,
					    1.0f,
					    timestep, 
					    slim.numNodes);

			calculateForces(doEdgeRepulsion, slim);
			
			program.getKernel("IntegrateRK3").execute(dimsGlobal, dimsLocal,
						bufferNodePosX, bufferNodePosY, 
					   	bufferNodeMass, 
					   	bufferNodeK,
					   	bufferNodeL,
					   	bufferVelocity,
					    bufferForce,
					    1.0f,
					    timestep, 
					    slim.numNodes);
		}
	
		/**
		 * Calculates all forces for the current state of the simulation
		 * and stores them for integration.
		 */
		private void calculateForces(boolean doEdgeRepulsion, SlimNetwork slim)
		{
				// Parallelization scheme is different for CPU and GPU kernel versions
				long[] dimsLocalEdgeRepulsion = new long[] { device.bestBlockSize };
				long[] dimsGlobalEdgeRepulsion = new long[] { Math.min(65536, nextMultipleOf(slim.numEdgesUnique, dimsLocalEdgeRepulsion[0])) };
				long[] dimsLocalGravity = new long[] { device.bestBlockSize };
				long[] dimsGlobalGravity = new long[] { device.type == DeviceTypes.GPU ? nextMultipleOf(slim.numNodes, dimsLocalGravity[0]) : slim.numNodesPadded / 2 };
				long[] dimsLocalSpring = device.type == DeviceTypes.GPU ? new long[] { 16, device.bestBlockSize / 16 } : new long[] { 1 };
				long[] dimsGlobalSpring = device.type == DeviceTypes.GPU ? new long[]{ 16, nextMultipleOf(slim.numNodes, dimsLocalSpring[1]) } : new long[] { slim.numNodes };
		
				if (device.type == DeviceTypes.GPU)
					program.getKernel("CalcForcesGravity").execute(dimsGlobalGravity, dimsLocalGravity,
						    new CyCLLocalSize(dimsLocalGravity[0] * 4), new CyCLLocalSize(dimsLocalGravity[0] * 4), new CyCLLocalSize(dimsLocalGravity[0] * 4),
						    bufferNodePosX, bufferNodePosY,
						    bufferNodeMass,
						    bufferForce,
						    slim.numNodes,
						    slim.numNodesPadded);
				else
					program.getKernel("CalcForcesGravity").execute(dimsGlobalGravity, dimsLocalGravity,
						    bufferNodePosX, bufferNodePosY,
						    bufferNodeMass,
						    bufferForce,
						    slim.numNodesPadded / 2);
				
				if (doEdgeRepulsion)
				{
					program.getKernel("PrepareEdgeRepulsion").execute(dimsGlobalEdgeRepulsion, dimsLocalEdgeRepulsion,
										bufferNodePosX, bufferNodePosY,
										bufferEdgeUniqueSources, bufferEdgeUniqueTargets,
										bufferEdgeStartX, bufferEdgeStartY,
										bufferEdgeTangentX, bufferEdgeTangentY,
										bufferEdgeCurrentLength,
										slim.numEdgesUnique);
					
					program.getKernel("CalcForcesEdgeRepulsion").execute(dimsGlobalGravity, dimsLocalGravity,
										new CyCLLocalSize(dimsLocalGravity[0] * 4), new CyCLLocalSize(dimsLocalGravity[0] * 4),	// position
										new CyCLLocalSize(dimsLocalGravity[0] * 4), new CyCLLocalSize(dimsLocalGravity[0] * 4),	// tangent
										new CyCLLocalSize(dimsLocalGravity[0] * 4),												// length
										new CyCLLocalSize(dimsLocalGravity[0] * 4), new CyCLLocalSize(dimsLocalGravity[0] * 4),	// mass
										bufferNodePosX, bufferNodePosY,
										bufferNodeMass,
										bufferEdgeStartX, bufferEdgeStartY,
										bufferEdgeTangentX, bufferEdgeTangentY,
										bufferEdgeCurrentLength, 
										bufferEdgeMassStart, bufferEdgeMassEnd,
										bufferForce,
										slim.numNodes,
										slim.numEdgesUniquePadded);
				}
				
				if (device.type == DeviceTypes.GPU)
					program.getKernel("CalcForcesSpringDrag").execute(dimsGlobalSpring, dimsLocalSpring,
									    new CyCLLocalSize(dimsLocalSpring[0] * dimsLocalSpring[1] * 2 * 4),
									    bufferNodePosX, bufferNodePosY, 
									    bufferEdges, bufferEdgeOffsets, bufferEdgeCounts,
									    bufferEdgeCoeffs, bufferEdgeLengths, 
									    bufferVelocity, 
									    bufferForce, 
									    slim.numNodes);
				else
					program.getKernel("CalcForcesSpringDrag").execute(dimsGlobalSpring, dimsLocalSpring,
									    bufferNodePosX, bufferNodePosY, 
									    bufferEdges, bufferEdgeOffsets, bufferEdgeCounts,
									    bufferEdgeCoeffs, bufferEdgeLengths, 
									    bufferVelocity, 
									    bufferForce, 
									    slim.numNodes);
		}
		
		private long nextMultipleOf(long n, long multipleOf)
		{
			return (n + multipleOf - 1) / multipleOf * multipleOf;
		}
	}
}
