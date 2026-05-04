import {StyleSheet, Text, View,TouchableOpacity, Button} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useState, useEffect, useRef } from 'react';
import { Alert } from 'react-native';
import * as FileSystem from "expo-file-system/legacy";
import * as Sharing from 'expo-sharing';
import { useUploadQueue } from '../contexts/UploadQueue';
//import { Video } from 'react-native-compressor';

// Same default as CameraScreen.js — both screens upload to the same HF Space.
const BACKEND_URL = "https://engdesfau26-smartphonepose-metrabs-server.hf.space";

export default function HomeScreen({navigation}) {
  const [video, setVideo] = useState(null);
  const queue = useUploadQueue();


  const pickImage = async () => {
    // No permissions request is necessary for launching the image library.
    // Manually request permissions for videos on iOS when `allowsEditing` is set to `false`
    // and `videoExportPreset` is `'Passthrough'` (the default), ideally before launching the picker
    // so the app users aren't surprised by a system dialog after picking a video.
    // See "Invoke permissions for videos" sub section for more details.
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission required', 'Permission to access the media library is required.');
      return;
    }

let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['videos'],
      allowsEditing: true,
      quality: 0,
      videoExportPreset: ImagePicker.VideoExportPreset.H264_960x540,
    });
    console.log(result);

    if (!result.canceled) {
      setVideo(result.assets[0]);
    }
  };

const downloadAndExportCSV = async (csvUrl) => {
  try {
    // Local path inside the app
    const localPath = FileSystem.documentDirectory + "pose.csv";

    // Download CSV from server
    const { uri } = await FileSystem.downloadAsync(csvUrl, localPath);
    console.log("CSV downloaded to:", uri);

    // Check if sharing is available
    const available = await Sharing.isAvailableAsync();
    if (!available) {
      Alert.alert("Sharing not available on this device");
      return;
    }

    // Open share dialog so user can export CSV
    await Sharing.shareAsync(uri, {
      mimeType: "text/csv",
      dialogTitle: "Export Pose CSV",
      UTI: "public.comma-separated-values-text", // iOS
    });

  } catch (error) {
    console.error("CSV download/export failed:", error);
    Alert.alert("Failed to export CSV");
  }
};
  const uploadVideo = async (videoFile) => {
    if (!videoFile) {
    Alert.alert("Please select a video first");
    return;
  }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);

    try {
/*
      console.log("Compressing video...");
      const compressedUri = await Video.compress(
        videoFile.uri,
        { compressionMethod: 'auto', maxScale: 1280 }
      );
      */
    const formData = new FormData();

    formData.append("file", {
      uri: videoFile.uri,
      name: "upload.mp4",
      type: "video/mp4",
    });

      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const data = await response.json();

      console.log(data);
      const csvUrl = BACKEND_URL + data.csv_url;
      console.log("Downloading CSV from:", csvUrl);
      await downloadAndExportCSV(csvUrl);
      Alert.alert("Upload successful!");
    } catch (error) {
      console.error(error);
      Alert.alert("Upload failed");
    }
  };

    return(
    <View style={styles.container}>
        <Text style={styles.title}>Smartphone Pose Estimation</Text>

            {/* Live queue status — shown when there's anything in flight */}
            {queue.inFlight > 0 && (
              <View style={styles.queueBanner}>
                <Text style={styles.queueBannerText}>
                  {queue.uploading ? '1 uploading' : ''}{queue.uploading && queue.pending.length ? ' · ' : ''}{queue.pending.length ? `${queue.pending.length} queued` : ''}
                </Text>
              </View>
            )}

            <TouchableOpacity style={styles.button} onPress={()=>navigation.navigate("Camera")}>
                <Text style={styles.buttonText}>Camera</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={pickImage}>
              <Text style={styles.buttonText}>Pick Video</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={()=>{uploadVideo(video)}}>
                <Text style={styles.buttonText}>Upload Video</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={()=>navigation.navigate("Files")}>
                <Text style={styles.buttonText}>Files</Text>
            </TouchableOpacity>
    </View>

    )
}
const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    flex: 1,
    paddingHorizontal: 20,
    backgroundColor: '#F5F7FA'
  },
    title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'gray',
    marginBottom: 80,
    color: '#1E1E1E'
  },
  button: {
    width: '100%',
    backgroundColor: 'gray',
    paddingVertical: 15,
    borderRadius: 8,
    marginBottom: 40,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    color: '#1E1E1E'
  },
  queueBanner: {
    width: '100%',
    backgroundColor: '#fff3cd',
    borderColor: '#ffd166',
    borderWidth: 1,
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 24,
  },
  queueBannerText: {
    color: '#664d03',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
});
