
import serial

COM="COM4"
BAUD=9600
SerialPort = serial.Serial(COM,BAUD,timeout=1)
OutgoingData ='Connected'
SerialPort.write(bytes(OutgoingData, 'utf-8'))

while (True):
    OutgoingData =input('>')
    SerialPort.write(bytes(OutgoingData, 'utf-8'))
