package org.cytoscape.opencl.layout;

import static org.cytoscape.work.ServiceProperties.*;

import java.util.Properties;

import org.cytoscape.cycl.CyCLDevice;
import org.cytoscape.cycl.CyCLFactory;
import org.cytoscape.service.util.AbstractCyActivator;
import org.cytoscape.view.layout.CyLayoutAlgorithm;
import org.cytoscape.work.undo.UndoSupport;
import org.osgi.framework.BundleContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CyActivator extends AbstractCyActivator {

	private final Logger logger = LoggerFactory.getLogger(CyActivator.class);
	
	public void start(BundleContext bc)  {	
		// Have to wait for the opencl-cycl bundle to start and register its CyCL service.
		registerServiceListener(bc, 
			(cycl, props) -> initialize(bc, cycl),
			(cycl, props) -> {}, 
			CyCLFactory.class);
	}
	
	
	private void initialize(BundleContext bc, CyCLFactory cycl) {
		new Thread(() -> {
			try {
				// Don't initialize if there are no OpenCL devices.
        if (!cycl.isInitialized()) {
					logger.error("OpenCL did not initialize. Cannot register '" + CLLayout.ALGORITHM_DISPLAY_NAME + "'.");
					return;
        }

        CyCLDevice device = cycl.getDevice(); // Get the best device
				
				UndoSupport undo = getService(bc, UndoSupport.class);

				CLLayout forceDirectedCLLayout = new CLLayout(undo, device);

		        Properties forceDirectedCLLayoutProps = new Properties();
		        forceDirectedCLLayoutProps.setProperty(PREFERRED_MENU, "Layout.Cytoscape Layouts");
		        forceDirectedCLLayoutProps.setProperty("preferredTaskManager", "menu");
		        forceDirectedCLLayoutProps.setProperty(TITLE, forceDirectedCLLayout.toString());
		        forceDirectedCLLayoutProps.setProperty(MENU_GRAVITY, "10.5");
				registerService(bc, forceDirectedCLLayout, CyLayoutAlgorithm.class, forceDirectedCLLayoutProps);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}).start();
	}
}

