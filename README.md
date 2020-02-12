<p align="center">
                                   <b>**Well tempered metadynamics analysis**</b>
</p>
If you are using this script please acknowledge me (Dr Owen Vickery) and cite the following DOI.

DOI: xxx

<p align="center">
                                   <b>**SCRIPT OVERVIEW**</b>
</p>

This script will allow you to analyse 2D well tempered metadynamics simulations

It is contains several functions which are controlled by a input file. 

This allows a reproducible workflow which can be released with the simulation raw data.

<p align="center">
                                   <b>**HILLS SORT**</b>
</p>

If you are running metadynamics with multiple walkers this function will sort the HILLS file according to deposition time.
Instead of the simulation timestamp.

The input file only requires the following variables to run this function.

- HILLS_sort = HILLS_sorted (output)
- HILLS = HILLS (input)
<p>
python metadynamics.py -f sort
</p>

<p align="center">
                                   <b>**HILLS SKIP**</b>
</p>

Analysis of the metadynamics HILLS file is difficult due to the number of entries (upto 100 million data points)

Therefore this section will read through the HILLS file and read out every 1 ns.

If no sorted HILLS file is found it will resort the original HILLS file.

The input file only requires the following variables to run this function.

- HILLS = HILLS (input)
- HILLS_sort = HILLS_sorted (output)
- HILLS_skip = HILLS_skipped (input)

<p>
python metadynamics.py -f skip
</p>

<p align="center">
                                   <b>**FES plot**</b>
</p>

This section will plot the 2d landscape

The following flags are required for plotting the 2d landscape:

- fes = fes.dat (input)
- CV1 = X (nm) 
- CV2 = Y (nm)
- energy_max = 20 (maximum FES height)
- step = 5 (contour level for FES)
- interval_x = 1 (x tick interval)
- interval_y = 1 (y tick interval)
- colour_bar_tick = 5 (colourbar tick interval)

Optional variables:

- bulk_values = bulk_area (file containing area to reference to zero)
- bulk_outline = True/False (plot dotted line around area referenced to bulk)

You can plot either circles or ellipses to highlight binding sites.

- ring_location = X,Y X,Y (each point separated by spaces)

- circle_plot = True/False
- circle_area = 0.5,0.6 (list of areas separated by commas)

- ellipse_plot = True/False
- ellipse_width = 3
- ellipse_height = 2
- ellipse_angle = 45  (degrees)

you can plot the picture of your protein under the energy landscape using the variables:

- picture = True/False 
- picture_file=extracellular/A2A_hills/picture.tga (file location of the picture to plot)
- picture_loc=2.7,-3.2, 3.1,-3.1 (the corners of the picture)


You can also plot the minimum free energy path between two point

- 1d = True/False
- search_width = 3 (with either side of the line between the points to search within)
- x_points =  0.0,5  (X coordinates for the start and end, if multiple ares need to be search enter them with a space between)
- y_points =  0,0  (Y coordinates for the start and end, if multiple ares need to be search enter them with a space between)
- 1d_location=1d_landscapes (location for the )
- stride = 0.25
