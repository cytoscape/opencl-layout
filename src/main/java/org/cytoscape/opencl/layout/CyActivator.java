package org.cytoscape.opencl.layout;

import static org.cytoscape.work.ServiceProperties.MENU_GRAVITY;
import static org.cytoscape.work.ServiceProperties.PREFERRED_MENU;
import static org.cytoscape.work.ServiceProperties.TITLE;

import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.cytoscape.application.CyApplicationConfiguration;
import org.cytoscape.opencl.cycl.CyCL;
import org.cytoscape.property.CyProperty;
import org.cytoscape.service.util.AbstractCyActivator;
import org.cytoscape.view.layout.CyLayoutAlgorithm;
import org.cytoscape.work.ServiceProperties;
import org.cytoscape.work.undo.UndoSupport;
import org.osgi.framework.BundleContext;

public class CyActivator extends AbstractCyActivator 
{
	public CyActivator() 
	{
		super();
	}

	public void start(BundleContext bc) 
	{	
		// Start OpenCL Layout in separate thread
		final ExecutorService service = Executors.newSingleThreadExecutor();
		service.submit(()-> {
			try {
				CyApplicationConfiguration applicationConfig = getService(bc, CyApplicationConfiguration.class);	
				CyProperty<Properties> cyPropertyServiceRef = getService(bc, CyProperty.class, "(cyPropertyName=cytoscape3.props)");
				
				CyCL.initialize(applicationConfig, cyPropertyServiceRef);
				
				// Don't initialize if there are no OpenCL devices.
				if (CyCL.getDevices().size() == 0)
					return;
				
				UndoSupport undo = getService(bc, UndoSupport.class);

				CLLayout forceDirectedCLLayout = new CLLayout(undo);

		        Properties forceDirectedCLLayoutProps = new Properties();
		        forceDirectedCLLayoutProps.setProperty(PREFERRED_MENU, "Layout.Cytoscape Layouts");
		        forceDirectedCLLayoutProps.setProperty("preferredTaskManager", "menu");
		        forceDirectedCLLayoutProps.setProperty(TITLE, forceDirectedCLLayout.toString());
		        forceDirectedCLLayoutProps.setProperty(MENU_GRAVITY, "10.5");
				registerService(bc, forceDirectedCLLayout, CyLayoutAlgorithm.class, forceDirectedCLLayoutProps);
			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}
}

