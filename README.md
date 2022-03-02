                                                        .`                 
                                                        /                  
                                                      `:`                  
                                                     .-`                   
                      `                             --                     
                      `-`                      `-`.-`                      
                       `-.                    --..o                        
                         `..`               .:.  `y.                       
    --`                    `-/-`          `/+`    -o`                      
     `...`                   `-/-`       `s+    ```-+`                     
         ..-`````               /s     `-+:  -/ossyyys+/-                  
            `-:://///:-.`      `yh    -o/`  :dmmmdhhhhyo+`                 
             ```......:+so-    :m/   oy/+o+:/yNNNmdhyhdo+o                 
            ``...........+hs.  .h.`.+dhmmhyydddNNddyshhhs/                 
           `.....-:/+/++++/os---+hhdmmddmdddmd/dmyyssyys+                  
          ``.-:/ohdhmdhhhhdy:dmmNmdddhyyysys+. sdyso+:. /-                 
          `.-+hhhmmmmmdhydmmNdmyhsos//ss/--`     `       o:                
           .+ydmdmmmmNmdddddd.-s+``o   -+++//-.          `/-```            
           .+sdmmmddddhoosyy+ s:   o/      ``..//-`         ``.......`     
            ./shdhhyyssyoo/. `d-   `y:          `-:-.``              `...  
              .-/+++//:-`    +s     .y.               `.-.              `  
                            :s`      -+                                    
                          `:-`        o-                                   
                        `::`          -+`                                  
                      `-:.             `-.                                 
                     .-`                 .-`                               
                    .`                    `:                               
                  `-`                      .`                              
                 `.       

# Accelerated Neutron Transport Solution (ANTS)

Accelerated Neutron Transport Solution (ANTS) calculates the neutron flux for both criticality and fixed source problems of one dimensional slabs and spheres and two dimensional slabs using the discrete ordinates method. It looks to combine machine learning with collision based hybrid methods and speedup through Numba, Cython, and C-functions.

### Spatial
- [ ] One Dimensional Slab
	- [ ] Diamond Difference
	- [ ] Step Method
	- [ ] Discontinuous Galerkin
- [ ] One Dimensional Sphere
	- [ ] Diamond Difference
	- [ ] Step Method
	- [ ] Discontinuous Galerkin
- [ ] Two Dimensional Slab
	- [ ] Diamond Difference
	- [ ] Step Method
	- [ ] Discontinuous Galerkin

### Time 
- [ ] Backward Euler (BDF1)
- [ ] BDF2
- [ ] TR-BDF2

### Acceleration
- [ ] DJINN
- [ ] Hybrid Method
- [ ] Synthetic Diffusion
- [ ] GMRES
- [ ] CMFD 

### Other
- [ ] Nearby Problems
