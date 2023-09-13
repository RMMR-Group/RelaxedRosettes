using RelaxedRosettes
using SignalAnalysis
using Plots
using Unitful

# generateTrajectory
(gOpt_with_spoiler, gOpt, target_points, rasters_between_points, gradient_raster, gradient_unit) =
    RelaxedRosettes.generateRosette(display_plots=true)

# save out the gradient as a binary file
# first save out the number of points in the gradient
# then save out each of the gradients, x, y, z
io = open("gradient.bin", "w")
write(io, Int32(size(gOpt_with_spoiler)[2]))
write(io, Int32(size(gOpt)[2]))
# write gradients to io as Float32
write(io, Float32.(gOpt_with_spoiler[1, :]))
write(io, Float32.(gOpt_with_spoiler[2, :]))
write(io, Float32.(gOpt_with_spoiler[3, :]))
close(io)

# get signals
gx_signal = signal(gOpt[1, :] * ustrip(gradient_unit), 1 / gradient_raster)
gy_signal = signal(gOpt[2, :] * ustrip(gradient_unit), 1 / gradient_raster)
gz_signal = signal(gOpt[3, :] * ustrip(gradient_unit), 1 / gradient_raster)

# create a new figure and plot power spectral density of the gradient
p1 = psd(gx_signal; yrange=[-80, 10], nfft=4096, label="gx")
psd!(gy_signal; yrange=[-80, 10], nfft=4096, label="gy")
psd!(gz_signal; yrange=[-80, 10], nfft=4096, label="gz")
xlims!(0, 2)
# vspan!(1:1, alpha=0.25, fillcolor=:red)

p2 = psd(sqrt.(gx_signal.^2 + gy_signal.^2 + gz_signal.^2); yrange=[-80, 0], nfft=4096, label="norm grad")
xlims!(0, 2)

plot(p1, p2, layout=(1, 2))
gui()
