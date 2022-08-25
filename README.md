# RK45_Orbit_Simulation
Simulates a comet around the sun. Is adaptable for any gravitational body.

## Verified with Comet Halley
Within the margin of error of the inputs, it does agree with observation:
* For period (70 vs 74 years)
* Aphelion - furthest distance from Sun - (33.4 vs 35.2 AU)

## Replayability
* Input any starting velocity or position
* Change the gravitational field

## First Code
First time I publish or code anything that works! \
Did not find a proper orbit simulator that uses the RK45 (Runge-Kutta-Fehlberg) method online. \
I hope it may help someone!

## Potential Improvements
Feel free to build on my code, reference it and keep it free!
* Making animations would be nice! Note that the time steps are variable
* Make it 3D
* Add more bodies and interactions between bodies

Note: System highly sensitive to initial conditions, example with 1% higher velocity in plot 
