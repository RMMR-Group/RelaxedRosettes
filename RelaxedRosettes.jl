using LinearAlgebra
using Random
using Distributions
using Unitful
using PaddedViews
using Convex
using ECOS
using FFTW
using GLMakie
using NFFT

function randCirclePoints(n)
	d = MvNormal([0,0,0], I);
	return hcat(map(x -> x ./ norm(x), eachcol(rand(d, n)))...)
end

function addNewBlueNoiseSpherePoint(ps)
    candidates = randCirclePoints(size(ps)[2] * 100)
    return [ps candidates[:,findmax(
        [minimum(norm.(eachcol(
            broadcast(-, reshape(x,(3,1)),ps)
            ))) for x in eachcol(candidates)]
        )[2]]]
end

function genrateNBlueNoiseSpherePoints(n)
	acceptedPoints = randCirclePoints(1)
	for i in 2:n
		acceptedPoints = addNewBlueNoiseSpherePoint(acceptedPoints)
	end
	return acceptedPoints
end
function sortSpherePoints(ps)
	orderedPoints = ps[:,1:1]
	remainingPoints = ps[:,2:end]
	while size(remainingPoints)[2] > 1
		sortedPoints = sortslices(
				remainingPoints,
				dims=2,
				by=(x->norm(x - orderedPoints[:,end])));
		farthestPoint = sortedPoints[:,end];
		orderedPoints = [orderedPoints farthestPoint]
		remainingPoints = sortedPoints[:,1:end-1];
	end
	return [orderedPoints remainingPoints[:,1]]
end

function generateTrajectory(numPoints, raster_between_targets)
	points = genrateNBlueNoiseSpherePoints(numPoints)
	sortedPoints = sortSpherePoints(points)

	d1 = Normal(0, 0.5);
	scales = abs.(1 .-abs.(rand(d1, size(sortedPoints)[2])))
	scales = scales ./ maximum(norm.(scales))
	scaledPoints = (sortedPoints' .* scales)'

	d2 = Normal(0, 0.2)
	scaled_leading_displacements = (rand(d2, size(scaledPoints))' .* scales)'
	scaled_trailing_displacements = (rand(d2, size(scaledPoints))' .* scales)'

	scaledLeadingRejections = scaled_leading_displacements .-
		(sum(scaled_leading_displacements .* scaledPoints, dims=(1)) ./ sum(scaledPoints .* scaledPoints, dims=(1))) .* scaledPoints

	scaledLeadingPoints =
		scaledPoints .* 0.5 +
		scaledLeadingRejections

	scaledTrailingPoints =
		scaledPoints .* 0.5 -
		scaledLeadingRejections
		
	target_points = reshape([zeros(size(sortedPoints))' scaledLeadingPoints' scaledPoints' scaledTrailingPoints']', 3, :)[:,1:end-1]

	rasters_between_points = hcat(
		round.(Int64,sqrt(scales[1]) .* [raster_between_targets * 2, raster_between_targets * 2, raster_between_targets, raster_between_targets]'),
		round.(Int64,(sqrt.(scales[2:end]) .* [raster_between_targets, raster_between_targets, raster_between_targets, raster_between_targets]')') ...)[1:end-2]

	g = Variable((size(target_points)[1], sum(rasters_between_points) + 1)...)

	pMat = hcat(map(i -> PaddedView(0,
			[ones(i)' 0.5]',
			(sum(rasters_between_points) + 1,),
			(1,)), [0 cumsum(rasters_between_points)']')...)

	slewMat = Tridiagonal(zeros(size(pMat)[1]-1), -ones(size(pMat)[1]), ones(size(pMat)[1]-1))[1:end-1,:]

	target_cons = g * pMat == target_points	

	problem = minimize(norm(vec(g * slewMat')), [target_cons])
	solve!(problem, ECOS.Optimizer)

	return (evaluate(g), target_points, rasters_between_points);
end

function runTest()

	k_fov = 1.0/(0.8u"mm")
	gradient_raster = 10.0u"µs"
	larmor_frequency = 42.58u"MHz/T"
	gradient_unit = uconvert(u"mT/m", k_fov / (larmor_frequency * gradient_raster))
	slew_unit = uconvert(u"T/m/s", gradient_unit / gradient_raster)

	(gOpt, target_points, rasters_between_points) = generateTrajectory(30, 75)

	print(uconvert.(u"ms", cumsum(rasters_between_points) .* gradient_raster)[1:2:end])

	f = Figure()
	ax1 = Axis3(f[1,1], aspect=(1,1,1))
	scatter!(ax1, target_points[:, 3:4:end] .* ustrip(k_fov), colormap = :batlow, color = cumsum(rasters_between_points)[3:4:end]);
	scatter!(ax1, target_points[:, 4:4:end] .* ustrip(k_fov), colormap = :batlow, color = cumsum(rasters_between_points)[4:4:end]);
	scatter!(ax1, target_points[:, 2:4:end] .* ustrip(k_fov), colormap = :batlow, color = cumsum(rasters_between_points)[2:4:end]);
	lines!(ax1, cumsum(gOpt,dims=2) .* ustrip(k_fov), colormap = :batlow, color = 1:size(gOpt)[2]);
	mesh!(ax1, Sphere(Point3f(0),1 .* ustrip(k_fov)), color = (:dodgerblue, 0.05))

	ax2 = Axis3(f[1,2], aspect=(1,1,1))
	lines!(ax2, gOpt .* ustrip(gradient_unit), colormap = :batlow, color = 1:size(gOpt)[2]);

	pMat = hcat(map(i -> PaddedView(0,
			[ones(i)' 0.5]',
			(sum(rasters_between_points) + 1,),
			(1,)), [0 cumsum(rasters_between_points)']')...)

	slewMat = Tridiagonal(zeros(size(pMat)[1]-1), -ones(size(pMat)[1]), ones(size(pMat)[1]-1))[1:end-1,:]

	ax3 = Axis3(f[1,3], aspect=(1,1,1))
	lines!(ax3, gOpt * slewMat' * ustrip(slew_unit), colormap = :batlow, color = 1:size(gOpt)[2]);

	gOptFreq = fftshift(fft(gOpt .* ustrip(gradient_unit), (2)), (2))
	ax4 = Axis(f[2,1])
	freqLims = ustrip(uconvert(u"Hz", 1.0/gradient_raster))/2.0
	freqBins = range(-freqLims, freqLims, length=size(gOptFreq)[2])
	lines!(ax4, freqBins, abs.(gOptFreq)[1,:])
	lines!(ax4, freqBins, abs.(gOptFreq)[2,:])
	lines!(ax4, freqBins, abs.(gOptFreq)[3,:])


	ax5 = Axis(f[2,2])
	test = mapslices(norm, gOptFreq, dims=1)
	lines!(ax5, freqBins, test[1,:])

	sampleKPoints = (cumsum(gOpt, dims=2) - gOpt ./ 2.0) ./ 2.0
	sampleTPoints = (2.0 * (cumsum(ones(size(gOpt)[2]), dims=2) .- 0.5) ./ size(gOpt)[2] .- 1.0) ./ 2.0
	print(size(sampleKPoints))
	print(size(sampleTPoints))
	sampleKTPoints = [sampleKPoints' sampleTPoints]'

	for i = 1:10
		(gOptTemp, target_points2, rasters_between_points2) = generateTrajectory(30, 75)
		newKPoints = (cumsum(gOptTemp, dims=2) - gOptTemp ./ 2.0) ./ 2.0
		sampleKPoints = [sampleKPoints newKPoints]
		newTPoints = (2.0 * (cumsum(ones(size(gOptTemp)[2]), dims=2) .- 0.5) ./ size(gOptTemp)[2] .- 1.0) ./ 2.0
		sampleKTPoints = [sampleKTPoints [newKPoints' newTPoints]' ]
	end

	print(size(sampleKPoints))

	testData = ones(ComplexF32, size(sampleKPoints)[2] )

	p3D = plan_nfft(sampleKPoints, (256, 256, 256))
	output3D = adjoint(p3D) * testData

	image(f[3,1], abs.(output3D[129,:,:]))

	#p4D = plan_nfft(sampleKTPoints, (10, 10, 10, 2000))
	#output4D = adjoint(p4D) * testData

	#image(f[3,2], abs.(output4D[17,17,:,:]))
	#image(f[3,3], abs.(output4D[17,:,:,1001]))

	f
	
end
