import numpy as np
import math
import random

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# grid formulation (100x100)
x_max = 100
y_max = 100
# diameter of wheel
wheel_d = 5
# north vector
north = [0, 1]
# delta t for continuous discretization
dt = 0.01
# mass of robot (an arbitrary value in units of kilograms)
m = 1.0
# volume of robot (an arbitrary value in units of cubic meters)
V = 0.1
# gravity constant (meters per seconds squared)
g = 9.8
# density of seawater (kilograms per cubic meter)
rho = 1023.6
# goal position
goal = [150, 150, -60]
# valve coefficient
cv = 10
# specific gravity of seawater
G = 1.025


# robot formulation
class Robot:
    def __init__(self, x, y, z, theta, phi=0, invivo_water=0, dx=0, dy=0, dz=0, dtheta=0, dphi=0, dwater=0):
        # angles in radians, invivo_water is a volume of water in the robot
        self.state = [x, y, z, theta, phi, invivo_water, dx, dy, dz, dtheta, dphi, dwater]
        #self.action = [rudder_PWM, propeller_PWM, IN_POWER, OUT_POWER]

    def pwmToRotVel(self, input_):
        output = np.array(100 * np.tanh(input_))
        # https://www.geeksforgeeks.org/numpy-tanh-python/ i used np beacuse of this, couldve been wrong though
        #output[0] = 100 * np.tanh(2 * r_vx)
        #output[1] = 100 * np.tanh(2 * r_vy)
        return output

    def propPwmToForce(self, power):
        ##TODO convert power delivered to the propeller to the thrust generated by it in dt
        return power
    
    def rudderPwmToDeltaAngle(self, power):
        ##TODO convert power delivered to the rudder to the change in angle of the rudder in dt
        return power

    def invalvePwrToDeltaWater(self, power):
        return

    def outvalvePwrToDeltaWater(self, power):
        return

    def updateState(self, nextState):
        #if nextState[0] > 0 and nextState[0] < 100:
            #self.state[0] = nextState[0]
        #if nextState[1] > 0 and nextState[1] < 100:
            #self.state[1] = nextState[1]
        #if nextState[2] > 0 and nextState[2] < 100:
            #self.state[2] = nextState[2]
        self.state = nextState

    def getNextState(self, input_):
        
        #rot_vel = self.pwmToRotVel(input_)

        #velocity = rot_vel*(wheel_d/2)

        #vbar = (velocity[0] + velocity[1]) / 2

        #system dynamics
        x,y,z,theta,phi,water,dx,dy,dz,dtheta,dphi,dwater = self.state
        rudder_pwr, prop_pwr, in_water = input_
        nState = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        prop_force = self.propPwmToForce(prop_pwr)
        #force_x = prop_force*math.cos(phi)*math.cos(theta) + prop_force*math.sin(phi)*math.sin(theta)
        #force_y = prop_force*math.cos(phi)*math.sin(theta) + prop_force*math.sin(phi)*math.cos(theta)
        force_x = prop_force*math.cos(theta)
        force_y = prop_force*math.sin(theta)
        buoyant_force = rho * g * V
        weight_force = g * (m + water * rho)
        force_z = buoyant_force - weight_force
        #x
        nState[0] = x + dx*dt + (force_x*dt*dt)/(2*m)
        #y
        nState[1] = y + dy*dt + (force_y*dt*dt)/(2*m)
        #z
        nState[2] = z + dz*dt + (force_z*dt*dt)/(2*m)
        if nState[2] > 0:
            nState[2] = 0
        #theta
        nState[3] = (theta - (self.rudderPwmToDeltaAngle(rudder_pwr) / 4)) % (2*math.pi)
        #phi
        nState[4] = phi + self.rudderPwmToDeltaAngle(rudder_pwr)
        if nState[4] > math.pi/2:
            nState[4] = math.pi/2
        if nState[4] < -math.pi/2:
            nState[4] = -math.pi/2
        #invivo_water
        nState[5] = water + in_water
        if nState[5] < 0:
            nState[5] = 0
        #dx
        nState[6] = dx + (force_x*dt)/m
        #dy
        nState[7] = dy + (force_y*dt)/m
        #dz
        nState[8] = dz + (force_z*dt)/m
        #dtheta
        nState[9] = self.rudderPwmToDeltaAngle(rudder_pwr) / 4
        #dphi
        nState[10] = self.rudderPwmToDeltaAngle(rudder_pwr)
        if nState[10] > math.pi/2:
            nState[10] = math.pi/2
        if nState[10] < -math.pi/2:
            nState[10] = -math.pi/2
        #dwater
        nState[11] = in_water

        return nState

    def state_dynamic_equation(self, input_, noise=None, time=None):
        # update state (member variables) from input_, noise, and time
        if len(input_) != 3:
            return "Please enter a four element input_! [rudder_PWM, propeller_PWM, in_water]"
        if noise:
            return "Haven't implemented noise yet, sorry! Please try again."
        if time:
            return "Ignoring time due to Markov property! Please try again."

        # get next state
        nextState = self.getNextState(input_)

        # Update State
        self.updateState(nextState)

    def output_equation(self, input_, noise=None, time=None):
        # return output as 5 dimensional vector from state (member variables), input_, noise, and time
        if len(input_) != 3:
            return "Please enter a four element input_! [rudder_PWM, propeller_PWM, in_water]"
        if noise:
            return "Haven't implemented noise yet, sorry! Please try again."
        if time:
            return "Ignoring time due to Markov property! Please try again."

        output_vec = [0] * 5

        x = self.state[0]
        y = self.state[1]
        angle = self.state[2]
        
        def getMainLineIntersection(x, y, angle):
            while angle >= 2*np.pi:
                angle -= 2*np.pi
            while angle < 0:
                angle += 2*np.pi
            if angle == 0:
                xwf = x_max
                ywf = y
            elif angle == np.pi/2:
                xwf = x
                ywf = y_max
            elif angle == np.pi:
                xwf = 0
                ywf = y
            elif angle == 3*np.pi/2:
                xwf = x
                ywf = 0
            else:
                slope = np.tan(angle)
                intercept = y-(slope*x)
                ywf = min(y_max,slope*x_max + intercept)
                ywf = max(ywf, 0)
                #ywb = slope*0 + intercept
                xwf = min(x_max,(y_max - intercept) / slope)
                xwf = max(xwf, 0)
                #xwb = (0 - intercept) / slope
            return (xwf,ywf)
            
        def getPerpLineIntersection(x, y, angle):
            angle -= np.pi/2
            while angle >= 2*np.pi:
                angle -= 2*np.pi
            while angle < 0:
                angle += 2*np.pi
            if angle == 0:
                xwr = x_max
                ywr = y
            elif angle == np.pi/2:
                xwr = x
                ywr = y_max
            elif angle == np.pi:
                xwr = 0
                ywr = y
            elif angle == 3*np.pi/2:
                xwr = x
                ywr = 0 
            else:
                slope = np.tan(angle)
                intercept = y-(slope*x)
                ywr = min(y_max, slope*x_max + intercept)
                ywr = max(ywr, 0)
                #ywl = slope*0 + intercept
                #xwl = (y_max - intercept) / slope
                xwr = min(x_max, (0 - intercept) / slope)
                xwr = max(xwr, 0)
            return (xwr,ywr)
        
        xwf,ywf = getMainLineIntersection(x,y,angle)
        xwr,ywr = getPerpLineIntersection(x,y,angle)
        output_vec[0] = np.sqrt((xwf - self.state[0]) ** 2 + (ywf - self.state[1]) ** 2)  # distance to the wall in front
        output_vec[1] = np.sqrt((xwr - self.state[0]) ** 2 + (ywr - self.state[1]) ** 2)  # distance to the wall to the right
        
        #output_vec[0] = np.linalg.norm(np.array([xwf,ywf]) - np.array([self.state[0],self.state[1]]))
        #output_vec[1] = np.linalg.norm(np.array([xwr,ywr]) - np.array([self.state[0],self.state[1]]))
        # convert PWM to rotational velocity
        rot_vel = self.pwmToRotVel(input_)
        velocity = rot_vel*(wheel_d/2)
        omega = velocity[0] - velocity[1]
        output_vec[2] = omega  # in plane rotational speed

        # take dot product of position vector with north, divide by their magnitudes, and take inverse cosine to get angle phi
        phi = np.arccos((self.state[0] * north[0] + self.state[1] * north[1]) / np.linalg.norm([self.state[0], self.state[1]]))
        output_vec[3] = np.cos(phi)  # magnetic field in x direction
        output_vec[4] = np.sin(phi)  # magnetic field in y direction

        return output_vec

# Test 1
rob = Robot(0, 0, 0, 0)
xs = []
ys = []
zs = []
thetas = []
phis = []
waters = []
dxs = []
dys = []
dzs = []
dthetas = []
dphis = []
dwaters = []
len_ = []
#print(rob.output_equation([0,0]))
for i in range(100):
    len_.append(i)
    rob.state_dynamic_equation([0,1,0.1])
    xs.append(rob.state[0])
    ys.append(rob.state[1])
    zs.append(rob.state[2])
    thetas.append(rob.state[3])
    phis.append(rob.state[4])
    waters.append(rob.state[5])
    dxs.append(rob.state[6])
    dys.append(rob.state[7])
    dzs.append(rob.state[8])
    dthetas.append(rob.state[9])
    dphis.append(rob.state[10])
    dwaters.append(rob.state[11])
#print(rob.output_equation([0,0]))
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#plt.title('Test 1')
plt.title("Test 1")
plt.show()
plt.plot(len_, xs, 'bo')
plt.title("x")
plt.show()
plt.plot(len_, ys, 'bo')
plt.title("y")
plt.show()
plt.plot(len_, zs, 'bo')
plt.title("z")
plt.show()
plt.plot(len_, thetas, 'bo')
plt.title("theta")
plt.show()
plt.plot(len_, phis, 'bo')
plt.title("phi")
plt.show()
plt.plot(len_, waters, 'bo')
plt.title("water")
plt.show()
plt.plot(len_, dxs, 'bo')
plt.title("dx")
plt.show()
plt.plot(len_, dys, 'bo')
plt.title("dy")
plt.show()
plt.plot(len_, dzs, 'bo')
plt.title("dz")
plt.show()
plt.plot(len_, dthetas, 'bo')
plt.title("dtheta")
plt.show()
plt.plot(len_, dphis, 'bo')
plt.title("dphi")
plt.show()
plt.plot(len_, dwaters, 'bo')
plt.title("dwater")
plt.show()

rob = Robot(0,0,0,math.pi/4)
xs = []
ys = []
zs = []
for i in range(100):
    rob.state_dynamic_equation([math.pi/8,1,.1])
    xs.append(rob.state[0])
    ys.append(rob.state[1])
    zs.append(rob.state[2])
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Test 2")
plt.show()


rob = Robot(0,0,0,math.pi/2)
xs = []
ys = []
zs = []
for i in range(100):
    rob.state_dynamic_equation([math.pi/2,1,.1])
    xs.append(rob.state[0])
    ys.append(rob.state[1])
    zs.append(rob.state[2])
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Test 3")
plt.show()


rob = Robot(200,200,-400,5*math.pi/4)
xs = []
ys = []
zs = []
for i in range(100):
    rob.state_dynamic_equation([math.pi/8,1,-.1])
    xs.append(rob.state[0])
    ys.append(rob.state[1])
    zs.append(rob.state[2])
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Test 4")
plt.show()


rob = Robot(100,100,-1000,7*math.pi/4)
xs = []
ys = []
zs = []
for i in range(100):
    rob.state_dynamic_equation([math.pi/4,1,-.1])
    xs.append(rob.state[0])
    ys.append(rob.state[1])
    zs.append(rob.state[2])
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Test 5")
plt.show()


rob = Robot(0,0,-2000,math.pi)
xs = []
ys = []
zs = []
for i in range(100):
    rob.state_dynamic_equation([0,0,-.1])
    xs.append(rob.state[0])
    ys.append(rob.state[1])
    zs.append(rob.state[2])
ax = plt.axes(projection='3d')
ax.plot3D(xs,ys,zs, 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Test 6")
plt.show()

#
##Test 2
#rob = Robot(80, 80, 0, 5*math.pi/4)
#xposes = []
#yposes = []
#zposes = []
#thetas = []
#phis = []
#waters = []
#len_ = []
##print(rob.output_equation([0,0]))
#for i in range(40):
    #len_.append(i)
    #rob.state_dynamic_equation([1,1,1])
    #xposes.append(rob.state[0])
    #yposes.append(rob.state[1])
    #zposes.append(rob.state[2])
    #thetas.append(rob.state[3])
    #phis.append(rob.state[4])
    #waters.append(rob.state[5])
##print(rob.output_equation([0,0]))
#ax = plt.axes(projection='3d')
#ax.plot3D(xposes,yposes,zposes)
#ax.set_xlabel('x pos')
#ax.set_ylabel('y pos')
#ax.set_zlabel('z pos')
#plt.title('Test 2')
#plt.show()
#plt.plot(len_, thetas)
#plt.title("thetas")
#plt.show()
#plt.plot(len_, phis)
#plt.title("phis")
#plt.show()
#plt.plot(len_, waters)
#plt.title("waters")
#plt.show()
#
##Test 3
#rob = Robot(20, 0, 0, math.pi/2)
#xposes = []
#yposes = []
#zposes = []
#thetas = []
#phis = []
#waters = []
#len_ = []
##print(rob.output_equation([0,0]))
#for i in range(40):
    #len_.append(i)
    #rob.state_dynamic_equation([1,1,1])
    #xposes.append(rob.state[0])
    #yposes.append(rob.state[1])
    #zposes.append(rob.state[2])
    #thetas.append(rob.state[3])
    #phis.append(rob.state[4])
    #waters.append(rob.state[5])
##print(rob.output_equation([0,0]))
#ax = plt.axes(projection='3d')
#ax.plot3D(xposes,yposes,zposes)
#ax.set_xlabel('x pos')
#ax.set_ylabel('y pos')
#ax.set_zlabel('z pos')
#plt.title('Test 3')
#plt.show()
#plt.plot(len_, thetas)
#plt.title("thetas")
#plt.show()
#plt.plot(len_, phis)
#plt.title("phis")
#plt.show()
#plt.plot(len_, waters)
#plt.title("waters")
#plt.show()