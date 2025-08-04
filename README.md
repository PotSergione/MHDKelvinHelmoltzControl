# Control of the Kelvin-Helmoltz mode 
Controls the KH mode in magnetised fluids using two coils outside of the simulation's domain. 
Below an image of the working controller:

![Demo](figures/mhd_controlled_0.001.gif)

![Demo](figures/mhd_uncontrolled_0.001.gif)


# Considerations to remember in the problem:
1) EFFECTIVE magnetic field: theory predicts that, for our intial conditions, a current of .3 ampere should be enough to generate a magnetic field 
capable of suppressing the instability from the infamus equivalence relation $$v_A*2 < Du$$. 

2) general units: nothing has been adimensionalised. We are taking into consideration incompressible, 2d mhd in the potential formulation. The velocity shear is .6 m/s, the velocity v far from the shear layer is .3 m/s.  The EFFECTIVE magnetic field, that is the value used in the computation of JxB for the vorticity equation, has been rescaled so that the magnetic field is actually sqrt{mu_0} B_0. All that has been discussed in the previous section refers to this rescaled field. This, in turn, implies that the actual magnetic field is in the order of the millitesla. The density is kept at 1kgm/m^3. 
Notice that the alfven speed in this setup is approx 1m/s, but it is still not costant along the domain due to the variations implied by the coils. 
The unit of time is therefore the second [s]. 

The reasonong is: in the case of a Helmoltz pair, separated along the common axis by a distance of R units, an arbitrary central field strength can be obtained through a current that satisfies a given mathematical formula. We employ this formula to have a baseline current that would generate a 1 tesla field at the axis, i.e the left wall of the domain. We then normalise the resulting field by multiplying it for $$\sqrt{\mu_0}$$ to enhance the robustness and simplify the threshold calculation. Spatial variations in the strength of the magnetic field imply that a stronger current than the one predicted by the theory for a uniform B must be specified. This current is equivalent, thanks to our specifications, to the magnetic field pre rescaling. The current is in perfect agreement with empirical findings concerning the expected decay of the field at increasing radii from the axis. 

## Boundary conditions
We have full dirichlet for the vorticity omega, neumann for A. Neumann for A require a bit of discussion. From a trivial computation is clear that this implies constant derivative for A along x. As A, by construction, mainly depends, up to negligible terms, on x, this implies that we are effectively requiring uniform A at the boundaries. In magnetic field terms, this is equivalent to stating that B must be exiting thedomain in the x direction at the boundaries. So this is conceptually different from a perfect conductor, that would have B 0 outside the domain. 