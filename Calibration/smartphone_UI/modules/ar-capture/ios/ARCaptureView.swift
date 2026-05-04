// ARCaptureView.swift
// Native ARView wrapper for Expo Modules API.
// Renders the ARKit camera feed as a React Native view component.

import ExpoModulesCore
import ARKit
import RealityKit

class ARCaptureView: ExpoView {

    private var arView: ARView?
    private var fallbackLabel: UILabel?

    required init(appContext: AppContext? = nil) {
        super.init(appContext: appContext)
        setupView()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupView() {
        backgroundColor = .black

        guard ARWorldTrackingConfiguration.isSupported else {
            let label = UILabel(frame: bounds)
            label.text = "ARKit requires a physical device"
            label.textAlignment = .center
            label.textColor = .white
            label.backgroundColor = .black
            label.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            addSubview(label)
            fallbackLabel = label
            return
        }

        let view = ARView(frame: bounds, cameraMode: .ar, automaticallyConfigureSession: false)
        view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        // Disable unnecessary RealityKit rendering features for better perf
        view.renderOptions = [.disablePersonOcclusion, .disableDepthOfField, .disableMotionBlur]
        addSubview(view)
        arView = view
    }

    func connectSession(_ session: ARSession) {
        arView?.session = session
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        arView?.frame = bounds
        fallbackLabel?.frame = bounds
    }
}
