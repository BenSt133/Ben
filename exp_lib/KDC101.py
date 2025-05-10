"""
To do:
- add functionality: moveTo, home, findMaxPower, setPower,...
- improve documentation
- improve error handling
"""
import clr
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from System import String
from System import Decimal
from System.Collections import *

# constants
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")

# Add references so Python can see .Net
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *

class KDC101:
    """Represents Thorlabs KDC101 brushed motor"""

    def __init__(self, serialNo = None):
        device_list_result = DeviceManagerCLI.BuildDeviceList()
        print("The device manager found the following devices:", \
            device_list_result)

        serial_numbers = DeviceManagerCLI.GetDeviceList(KCubeDCServo.DevicePrefix)
        print("The devices with the following serial numbers are connected:", serial_numbers)

        if serialNo == None:
            try:
                serialNo = serial_numbers[0]
            except IndexError:
                print("Cannot connect if no serial numbers found.")

        print("Attempt to connect to", serialNo)

        try:
            self.instrument = KCubeDCServo.CreateKCubeDCServo(serialNo)
            self.instrument.Connect(serialNo)
            print("Successfully connected to", serialNo)
        except:
            print("Could not connect. Try restarting the motor.")

        # Start the device polling
        # The polling loop requests regular status requests to the motor to ensure the program keeps track of the device
        self.instrument.StartPolling(250)
        # Needs a delay so that the current enabled state can be obtained
        time.sleep(0.5)
        # Enable the channel otherwise any move is ignored 
        self.instrument.EnableDevice()
        # Needs a delay to give time for the device to be enabled
        time.sleep(0.5)

        # Call LoadMotorConfiguration on the device to initialize the DeviceUnitConverter object required for real world unit parameters
        # - loads configuration information into channel
        motorConfiguration = self.instrument.LoadMotorConfiguration(serialNo)
        print(motorConfiguration)

        # The API requires stage type to be specified
        motorConfiguration.DeviceSettingsName = "PRM1Z8"

        # Get the device unit converter
        motorConfiguration.UpdateCurrentConfiguration()

        # Not used directly in example but illustrates how to obtain device settings
        currentDeviceSettings = self.instrument.MotorDeviceSettings

        # Updates the motor controller with the selected settings
        self.instrument.SetSettings(currentDeviceSettings, True, False)

        # Display info about device
        deviceInfo = self.instrument.GetDeviceInfo()

        if not self.instrument.Status.IsHomed:
            print("Device is not homed. Call KDC101.home() to home")

    def disconnect(self):
        self.instrument.StopPolling()
        self.instrument.Disconnect(True)
        print("Successfully disconnected.")

    def getPosition(self):
        return float(str(self.instrument.Position))

    def moveto(self, angle):
        self.instrument.SetMoveAbsolutePosition(Decimal(angle))
        self.instrument.MoveAbsolute(60000)

    def moveRelative(self, mag, dir = 1):
        '''
        Rotates the motor by mag degrees in counterclockwise (dir = 1)
        or counterclockwise directions (dir = -1)
        Returns new position.
        '''
        print("Starting to move at ", self.instrument.Position)
        self.instrument.SetMoveRelativeDistance(Decimal(dir*mag))
        self.instrument.MoveRelative(60000)
        print("Moved the device to ", self.instrument.Position)

        return float(str(self.instrument.Position))

    def setVelocity(self, velocity, acceleration = 10):   
        self.instrument.SetVelocityParams(Decimal(velocity), Decimal(acceleration))
        print("Changed velocity to", velocity, "and acceleration to", acceleration)

    def fullSweep(self, pm):
        '''
        Does a full 360 degree turn of the motor and records the
        power on the PM100USB as a function of the half-wave plate angle.
        
        Inputs:
        -------------
        pm      powermeter object with pm.power method
        '''
        # sweep up and take data
        pos_list = []
        power_list = []
        # polling allows us to determin position on the fly
        # requests status update every [input] millseconds
        self.instrument.StartPolling(50)
        print("Starting to move at ", self.instrument.Position)
        self.instrument.SetMoveRelativeDistance(Decimal(370))
        self.instrument.MoveRelative(0)
        time.sleep(0.3)
        
        # anything larger than 360 works for initial pos
        pos = 1000
        # while moving, take data
        while pos != float(str(self.instrument.Position)):
            # take data from another instrument
            power = pm.power
            pos = float(str(self.instrument.Position)) # convert system decimal to float
            # print("Position:", pos, "Power", power)
            pos_list.append(pos)
            power_list.append(power)
            time.sleep(0.1)  # delay will determin number of datapoints
            # print(float(str(self.instrument.Position)))
        self.instrument.StopPolling()
        print("Sweep done. Position: ", self.instrument.Position)
        
        pos_arr = np.array(pos_list)
        power_arr = np.array(power_list)
        time.sleep(2)

        plt.plot(pos_arr, power_arr)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Power (W/cm^2)')
        plt.show()

        plt.plot(pos_arr)  # plot to check position samples are linear
        plt.ylabel('Angle (deg)')
        plt.xlabel('time')
        plt.show()
        
        return pos_arr, power_arr

    def direction_of_increasing_power(self, pm):
        # not tested
        """
        returns the direction of increasing power 
        (1 = counterclockwise, -1 = clockwise)
        """
        p0 = pm.power
        self.moveRelative(1,1)
        p1 = pm.power
        self.moveRelative(1,-1)
        return np.sign(p1-p0)

    def direction_of_weakest_change(self, pm):
        # not tested
        """
        returns the direction of weakest change in power and the direction of
        increasing power (1 = counterclockwise, -1 = clockwise)
        """
        p0 = pm.power
        self.moveRelative(1,1)
        p1 = pm.power
        self.moveRelative(2,-1)
        pm1 = pm.power
        self.moveRelative(1,1)
        return np.sign(np.abs(p1-p0) - np.abs(pm1-p0)), np.sign(p1-pm1)

    def find_maximum(self, pm):
        # not tested
        dir = self.direction_of_increasing_power(pm)
        p1 = pm.power

        # make this a nicer for loop
        while True:
            self.moveRelative(1,dir)
            p0 = p1
            p1 = pm.power
            if(p0 > p1):
                break

        while True:
            self.moveRelative(0.1,-dir)
            p0 = p1
            p1 = pm.power
            if(p0 > p1):
                break

    def home(self):
        self.instrument.Home()

        
            


    


    

    

        