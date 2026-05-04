import { Button, StyleSheet, Text, TouchableOpacity, View} from 'react-native';
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import HomeScreen from './screens/HomeScreen';
import CameraScreen from './screens/CameraScreen';
import NavigationScreen from './screens/NavigationScreen';
import FilesScreen from './screens/FilesScreen';
import { UploadQueueProvider } from './contexts/UploadQueue';

const Stack = createNativeStackNavigator();

export default function App() {
  return(
  <UploadQueueProvider>
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name ='Home' component={HomeScreen}/>
        <Stack.Screen name ='Camera' component={CameraScreen}/>
        <Stack.Screen name ='Navigation' component={NavigationScreen}/>
        <Stack.Screen name ='Files' component={FilesScreen} options={{ headerShown: false }}/>
      </Stack.Navigator>
    </NavigationContainer>
  </UploadQueueProvider>

  );
}
