#define m11 3
#define m12 4
#define m21 5
#define m22 6
#include <LiquidCrystal_I2C.h>

 // set the LCD address to 0x3F for a 16 chars and 2 line display
 LiquidCrystal_I2C lcd(0x27,16,2); 
void setup()
{
  pinMode(m11, OUTPUT);
  pinMode(m12, OUTPUT);
  pinMode(m21, OUTPUT);
  pinMode(m22, OUTPUT);
  Serial.begin(9600);
 
  lcd.init();
  lcd.clear();         
  lcd.backlight();      // Make sure backlight is on
  
  // Print a message on both lines of the LCD.
  lcd.setCursor(2,0);   //Set cursor to character 2 on line 0
  lcd.print("Rover Ready!");
  
  lcd.setCursor(2,1);   //Move cursor to character 2 on line 1
  lcd.print("Lets Begin! :)");
}



void loop()
{
 
  while(Serial.available())
  {
    char In=Serial.read();
    Serial.print(In);

    if (In=='A')
    {
      Serial.print("Speed limit (20km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(20km/h)");
    }
    else if (In=='B')
    {
      Serial.print("Speed limit (30km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(30km/h)");
    }
     else if (In=='C')
    {
      Serial.print("Speed limit (50km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(50km/h)");
    }
    else if (In=='D')
    {
      Serial.print("Speed limit (60km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(60km/h)");
    }
    
    else if (In=='E')
    {
      Serial.print("Speed limit (70km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(70km/h)");
    }

    else if (In=='F')
    {
      Serial.print("Speed limit (80km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(80km/h)");
    }
    else if (In=='G')
    {
      Serial.print("End of Speed limit (80km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("End of Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(80km/h)");
    }

    else if (In=='H')
    {
      Serial.print("Speed limit (100km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(100km/h)");
    }
    else if (In=='I')
    {
      Serial.print("Speed limit (120km/h)\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Speed limit");
      lcd.setCursor(2,1);
      lcd.print("(120km/h)");
    }
    else if (In=='J')
    {
      Serial.print("No passing\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("NO");
      lcd.setCursor(2,1);
      lcd.print("Passsing");
    }
    
    else if (In=='L')
    {
      Serial.print("Right-of-way at the next intersection\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Right-of-way");
      lcd.setCursor(2,1);
      lcd.print("at intersection");
    }
    else if (In=='M')
    {
      Serial.print("Priority road\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Priority");
      lcd.setCursor(2,1);
      lcd.print("road");
    }
    else if (In=='N')
    {
      Serial.print("Yield\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Yield");
      lcd.setCursor(2,1);
      lcd.print(" ");
    }
    else if (In=='O')
    {
      Serial.print("Stop");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("STOP");
      lcd.setCursor(2,1);
      lcd.print(" ");
      digitalWrite(m11, LOW);
      digitalWrite(m12, LOW);
      digitalWrite(m21, LOW);
      digitalWrite(m22, LOW);
      
    }
     else if (In=='P')
    {
      Serial.print("No vehicles\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("No ");
      lcd.setCursor(2,1);
      lcd.print("Vehicles");
    }
    else if (In=='Q' || In=='K')
    {
      Serial.print("Vehicles over 3.5 metric tons prohibited");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("No 3.5 metric");
      lcd.setCursor(2,1);
      lcd.print("tons allowed");
    }
    
    else if (In=='R')
    {
      Serial.print("No entry");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("NO");
      lcd.setCursor(2,1);
      lcd.print("ENTRY");
      digitalWrite(m11, LOW);
      digitalWrite(m12, LOW);
      digitalWrite(m21, LOW);
      digitalWrite(m22, LOW);
    }

    else if (In=='S')
    {
      Serial.print("General caution\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("General");
      lcd.setCursor(2,1);
      lcd.print("caution");
    }
    else if (In=='T')
    {
      Serial.print("Dangerous curve to the left");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Dangerous curve ");
      lcd.setCursor(2,1);
      lcd.print("LEFT");
    }

    else if (In=='U')
    {
      Serial.print("Dangerous curve to the right\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Dangerous curve");
      lcd.setCursor(2,1);
      lcd.print("Right");
    }
    else if (In=='V')
    {
      Serial.print("Double Curve\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Double");
      lcd.setCursor(2,1);
      lcd.print("curve");
    }
    else if (In=='W')
    {
      Serial.print("Bumpy Road\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Bumpy");
      lcd.setCursor(2,1);
      lcd.print("Road");
    }
    else if (In=='X')
    {
      Serial.print("Slippery road\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Slippery");
      lcd.setCursor(2,1);
      lcd.print("road");
    }
    else if (In=='Y')
    {
      Serial.print("Road narrows on the right\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Road narrows");
      lcd.setCursor(2,1);
      lcd.print("Right");
    }
    else if (In=='Z')
    {
      Serial.print("Road work\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Road");
      lcd.setCursor(2,1);
      lcd.print("Work");
    }

    else if (In=='a')
    {
      Serial.print("Traffic signals\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Traffic");
      lcd.setCursor(2,1);
      lcd.print("Signals");
    }
    else if (In=='b')
    {
      Serial.print("Pedestrians\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Pedestrians");
      lcd.setCursor(2,1);
      lcd.print("Crossing");
    }
     else if (In=='c')
    {
      Serial.print("Children Crossing\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Children");
      lcd.setCursor(2,1);
      lcd.print("Crossing)");
    }
    else if (In=='d')
    {
      Serial.print("Bicyle Crossing\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Bicycle");
      lcd.setCursor(2,1);
      lcd.print("Crossing");
    }
    
    else if (In=='e')
    {
      Serial.print("Beware of ice/snow\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Beware of");
      lcd.setCursor(2,1);
      lcd.print("ice/snow");
    }

    else if (In=='f')
    {
      Serial.print("Wild animals crossing");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Wild Animals");
      lcd.setCursor(2,1);
      lcd.print("Crossing");
    }
    else if (In=='g')
    {
      Serial.print("End of all speed and passing limits\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("End of");
      lcd.setCursor(2,1);
      lcd.print("limits");
    }

    else if (In=='h')
    {
      Serial.print("Turn Right Ahead\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Turn Right");
      lcd.setCursor(2,1);
      lcd.print("Ahead");
      digitalWrite(m11, LOW);
      digitalWrite(m12, LOW);
      digitalWrite(m21, HIGH);
      digitalWrite(m22, LOW);
      
    }
    else if (In=='i')
    {
      Serial.print("Turn Left Ahead\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Turn Left");
      lcd.setCursor(2,1);
      lcd.print("Ahead");
       digitalWrite(m11, HIGH);
      digitalWrite(m12, LOW);
      digitalWrite(m21, LOW);
      digitalWrite(m22, LOW);
    }
    else if (In=='j')
    {
      Serial.print("Ahead only\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Ahead");
      lcd.setCursor(2,1);
      lcd.print("Only");
      digitalWrite(m11, HIGH);
      digitalWrite(m12, LOW);
      digitalWrite(m21, HIGH);
      digitalWrite(m22, LOW);
      
    }
    else if (In=='k')
    {
      Serial.print("Go straight or right");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Go straight");
      lcd.setCursor(2,1);
      lcd.print("or right");
    }
    else if (In=='l')
    {
      Serial.print("Go straight or left\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Go straight");
      lcd.setCursor(2,1);
      lcd.print("or left");
    }
    else if (In=='m')
    {
      Serial.print("Keep Right\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Keep");
      lcd.setCursor(2,1);
      lcd.print("Right");
    }
   else if (In=='n')
    {
      Serial.print("Keep Left\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Keep");
      lcd.setCursor(2,1);
      lcd.print("Left");
    }
    else if (In=='o')
    {
      Serial.print("Roundabout mandatory\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Roundabout");
      lcd.setCursor(2,1);
      lcd.print("mandatory");
    }
     else if (In=='p')
    {
      Serial.print("End of no passing\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("End of");
      lcd.setCursor(2,1);
      lcd.print("no passing");
    }
    else if (In=='q')
    {
      Serial.print("Ending of no passing by vechiles over 3.5 metric tons\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("3.5 MT vehicles");
      lcd.setCursor(2,1);
      lcd.print("allowed");
    }
 else
 {
      Serial.print("Unknown\n");
      lcd.clear();   
      lcd.setCursor(2,0);
      lcd.print("Connected");
      lcd.setCursor(2,1);
      lcd.print("Waiting...");
    }
  }
}
