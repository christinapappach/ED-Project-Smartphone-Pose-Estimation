// FilesScreen.js — browse / share / delete recordings.
// Each recording lives in its own folder under the app's Documents directory,
// e.g. recording_<unix_ts>/{video.mp4, depth.bin, intrinsics.json, output.csv}.
// This screen lists every such folder, plus any orphan files at the top level
// (left over from older builds), and lets the user share or delete each item.

import { View, Text, FlatList, TouchableOpacity, Alert, StyleSheet } from 'react-native';
import { useState, useCallback } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import * as FileSystem from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';

const MIME_BY_EXT = {
  '.mp4': 'video/mp4',
  '.csv': 'text/csv',
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
};

function mimeFor(name) {
  const lower = name.toLowerCase();
  for (const ext of Object.keys(MIME_BY_EXT)) {
    if (lower.endsWith(ext)) return MIME_BY_EXT[ext];
  }
  return 'application/octet-stream';
}

function formatBytes(b) {
  if (!b || b < 1024) return `${b || 0} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatTime(ts) {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleString();
}

export default function FilesScreen({ navigation }) {
  // recordings: array of { name, uri, isDirectory, modificationTime, size, files? }
  // files (when present): array of inner files
  // orphans: flat files at Documents root not inside a recording folder
  const [recordings, setRecordings] = useState([]);
  const [orphans, setOrphans] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadFiles = useCallback(async () => {
    setLoading(true);
    try {
      const dir = FileSystem.documentDirectory;
      const names = await FileSystem.readDirectoryAsync(dir);
      const recs = [];
      const orph = [];

      for (const name of names) {
        const uri = dir + name;
        let info;
        try { info = await FileSystem.getInfoAsync(uri); } catch { continue; }
        if (!info.exists) continue;

        if (info.isDirectory) {
          // Read inner files.
          let innerNames = [];
          try { innerNames = await FileSystem.readDirectoryAsync(uri + '/'); } catch {}
          const innerFiles = await Promise.all(innerNames.map(async (innerName) => {
            const innerUri = `${uri}/${innerName}`;
            let innerInfo = { size: 0, modificationTime: 0 };
            try { innerInfo = await FileSystem.getInfoAsync(innerUri); } catch {}
            return {
              name: innerName,
              uri: innerUri,
              size: innerInfo.size || 0,
              modificationTime: innerInfo.modificationTime || 0,
              mime: mimeFor(innerName),
            };
          }));
          const totalSize = innerFiles.reduce((a, f) => a + (f.size || 0), 0);
          recs.push({
            name,
            uri,
            isDirectory: true,
            modificationTime: info.modificationTime || 0,
            size: totalSize,
            files: innerFiles.sort((a, b) => a.name.localeCompare(b.name)),
          });
        } else {
          orph.push({
            name,
            uri,
            isDirectory: false,
            modificationTime: info.modificationTime || 0,
            size: info.size || 0,
            mime: mimeFor(name),
          });
        }
      }

      recs.sort((a, b) => b.modificationTime - a.modificationTime);
      orph.sort((a, b) => b.modificationTime - a.modificationTime);
      setRecordings(recs);
      setOrphans(orph);
    } catch (err) {
      console.log('Failed to load files:', err);
      Alert.alert('Error', err.message || 'Could not read documents directory');
    }
    setLoading(false);
  }, []);

  useFocusEffect(useCallback(() => {
    loadFiles();
  }, [loadFiles]));

  const handleShareFile = async (file) => {
    try {
      const available = await Sharing.isAvailableAsync();
      if (!available) {
        Alert.alert('Sharing not available on this device');
        return;
      }
      await Sharing.shareAsync(file.uri, {
        mimeType: file.mime,
        dialogTitle: file.name,
      });
    } catch (err) {
      Alert.alert('Share failed', err.message || String(err));
    }
  };

  const handleDeleteFile = (file, onAfter) => {
    Alert.alert(
      'Delete file?',
      `${file.name}\n${formatBytes(file.size)}\n\nThis can't be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await FileSystem.deleteAsync(file.uri, { idempotent: true });
              if (onAfter) onAfter();
              await loadFiles();
            } catch (err) {
              Alert.alert('Delete failed', err.message || String(err));
            }
          },
        },
      ]
    );
  };

  const handleDeleteRecording = (rec) => {
    Alert.alert(
      'Delete recording?',
      `${rec.name}\n${rec.files.length} files · ${formatBytes(rec.size)}\n\nThis deletes the folder and everything inside.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await FileSystem.deleteAsync(rec.uri, { idempotent: true });
              await loadFiles();
            } catch (err) {
              Alert.alert('Delete failed', err.message || String(err));
            }
          },
        },
      ]
    );
  };

  const renderRecording = ({ item: rec }) => (
    <View style={styles.recordingCard}>
      <View style={styles.recordingHeader}>
        <Text style={styles.recordingName} numberOfLines={1}>{rec.name}</Text>
        <Text style={styles.recordingMeta}>
          {rec.files.length} files · {formatBytes(rec.size)} · {formatTime(rec.modificationTime)}
        </Text>
      </View>

      {rec.files.map((file) => (
        <View key={file.uri} style={styles.fileRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.fileName} numberOfLines={1}>{file.name}</Text>
            <Text style={styles.fileMeta}>{formatBytes(file.size)}</Text>
          </View>
          <TouchableOpacity onPress={() => handleShareFile(file)} style={[styles.smallBtn, styles.shareBtn]}>
            <Text style={styles.smallBtnText}>Share</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => handleDeleteFile(file)} style={[styles.smallBtn, styles.deleteBtn]}>
            <Text style={styles.smallBtnText}>Delete</Text>
          </TouchableOpacity>
        </View>
      ))}

      <View style={styles.recordingActions}>
        <TouchableOpacity onPress={() => handleDeleteRecording(rec)} style={[styles.actionBtn, styles.deleteBtn]}>
          <Text style={styles.actionText}>Delete entire recording</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderOrphan = ({ item: file }) => (
    <View style={styles.orphanRow}>
      <View style={{ flex: 1 }}>
        <Text style={styles.fileName} numberOfLines={1}>{file.name}</Text>
        <Text style={styles.fileMeta}>{formatBytes(file.size)} · {formatTime(file.modificationTime)}</Text>
      </View>
      <TouchableOpacity onPress={() => handleShareFile(file)} style={[styles.smallBtn, styles.shareBtn]}>
        <Text style={styles.smallBtnText}>Share</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => handleDeleteFile(file)} style={[styles.smallBtn, styles.deleteBtn]}>
        <Text style={styles.smallBtnText}>Delete</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Text style={styles.backArrow}>‹</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Files</Text>
        <Text style={styles.countBadge}>{recordings.length}</Text>
        <TouchableOpacity onPress={loadFiles} style={styles.refreshBtn}>
          <Text style={styles.refreshText}>Refresh</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={recordings}
        keyExtractor={(item) => item.uri}
        contentContainerStyle={recordings.length === 0 && orphans.length === 0 && styles.emptyContainer}
        renderItem={renderRecording}
        ListFooterComponent={
          orphans.length > 0 ? (
            <View style={{ marginTop: 16 }}>
              <Text style={styles.sectionHeader}>Loose files (older builds)</Text>
              {orphans.map((f) => (
                <View key={f.uri}>{renderOrphan({ item: f })}</View>
              ))}
            </View>
          ) : null
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>
              {loading ? 'Loading…' : 'No recordings yet.'}
            </Text>
            {!loading && (
              <Text style={styles.emptyHint}>
                Record a clip from the Camera screen — each recording will appear here as a folder containing the video, depth file, intrinsics JSON, and (after upload) the output CSV.
              </Text>
            )}
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  header: {
    paddingTop: 56, paddingBottom: 12, paddingHorizontal: 16,
    flexDirection: 'row', alignItems: 'center',
    borderBottomWidth: 1, borderBottomColor: '#222',
  },
  backBtn: { paddingHorizontal: 4, paddingVertical: 2 },
  backArrow: { color: '#0a84ff', fontSize: 28, lineHeight: 28 },
  title: { color: '#fff', fontSize: 22, fontWeight: '700', marginLeft: 8 },
  countBadge: {
    color: '#888', fontSize: 14, marginLeft: 8, paddingHorizontal: 8, paddingVertical: 2,
    backgroundColor: '#1c1c1e', borderRadius: 10,
  },
  refreshBtn: { marginLeft: 'auto', paddingHorizontal: 8, paddingVertical: 4 },
  refreshText: { color: '#0a84ff', fontSize: 15, fontWeight: '600' },
  sectionHeader: {
    color: '#888', fontSize: 12, fontWeight: '600',
    marginHorizontal: 16, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5,
  },
  recordingCard: {
    marginHorizontal: 12, marginTop: 8, marginBottom: 8,
    padding: 12, backgroundColor: '#1c1c1e', borderRadius: 10,
  },
  recordingHeader: { marginBottom: 8 },
  recordingName: { color: '#fff', fontSize: 15, fontWeight: '700' },
  recordingMeta: { color: '#888', fontSize: 12, marginTop: 2 },
  fileRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingVertical: 8, borderTopWidth: 1, borderTopColor: '#2c2c2e',
  },
  fileName: { color: '#eee', fontSize: 13, fontWeight: '500' },
  fileMeta: { color: '#666', fontSize: 11, marginTop: 2 },
  smallBtn: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 6, marginLeft: 6 },
  smallBtnText: { color: '#fff', fontSize: 12, fontWeight: '600' },
  shareBtn: { backgroundColor: '#0a84ff' },
  deleteBtn: { backgroundColor: '#ff3b30' },
  recordingActions: { flexDirection: 'row', marginTop: 10 },
  actionBtn: { paddingVertical: 7, paddingHorizontal: 12, borderRadius: 7, marginRight: 8 },
  actionText: { color: '#fff', fontSize: 13, fontWeight: '600' },
  orphanRow: {
    flexDirection: 'row', alignItems: 'center',
    marginHorizontal: 12, marginBottom: 6, padding: 12,
    backgroundColor: '#1c1c1e', borderRadius: 10,
  },
  emptyContainer: { flex: 1, justifyContent: 'center' },
  empty: { alignItems: 'center', paddingHorizontal: 40 },
  emptyText: { color: '#666', fontSize: 16 },
  emptyHint: { color: '#444', fontSize: 13, marginTop: 8, textAlign: 'center', lineHeight: 18 },
});
