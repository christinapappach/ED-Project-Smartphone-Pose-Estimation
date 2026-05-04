// HomeScreen.js — unchanged from teammate's original
import {StyleSheet, Text, View, Button} from 'react-native';

export default function HomeScreen({navigation}) {
  return(
    <View>
      <Text>Home Screen</Text>
      <Button title="Go to Camera" onPress={()=>navigation.navigate("Camera")}></Button>
    </View>
  )
}
