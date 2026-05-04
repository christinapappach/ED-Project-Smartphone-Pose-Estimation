require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'ARCapture'
  s.version        = package['version']
  s.summary        = 'ARKit capture module for Expo'
  s.description    = 'Native ARKit capture with video recording, pitch tracking, and MeTRAbs upload'
  s.license        = 'MIT'
  s.author         = 'Christina Pappachan'
  s.homepage       = 'https://github.com/example'
  s.platforms      = { :ios => '15.1' }
  s.source         = { :git => 'https://github.com/example.git' }
  s.static_framework = true
  s.source_files   = '**/*.swift'

  s.dependency 'ExpoModulesCore'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_COMPILATION_MODE' => 'wholemodule'
  }
end
