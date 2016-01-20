Math.radians = function(degrees) {
  return degrees * Math.PI / 180;
};

function get_ellipse_coords(a, b, x, y, angle, num_pics, k=10){

	xs = [];
	ys = [];

	// beta = -angle*Math.PI/180.0;
	beta = angle;
	sin_beta = Math.sin(beta);
	cos_beta = Math.cos(beta);
	
	for(var theta=0; theta < (361); theta+=k){
		
		alpha = Math.radians(theta);
		sin_alpha = Math.sin(alpha);
		cos_alpha = Math.cos(alpha);

		xs.push(x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta));
		ys.push(y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta));
	}

	// calculate density (1e-6 is the minimum area)
	var d = num_pics/(a*b)/1e6;
	d = d>0.9 ? 0.9 : d;
		
	return [xs, ys, d];
}


function show_polygon(map, xs, ys, cluster_info, density=0.5, color='#FF0000'){

	// Define the LatLng coordinates for the polygon.
	var triangleCoords = [];

	for(var i=0; i<xs.length; i+=1){
		triangleCoords.push({lat: ys[i], lng: xs[i]});
	}

	// Construct the polygon.
	var polygon = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: color,
		strokeOpacity: 1.0,
		strokeWeight: 2,
		fillColor: color,
		fillOpacity: density
	});
	polygon.setMap(map);
	
	// Add a listener for the click event.
	polygon.addListener('click', function (event){
		showArrays(event, cluster_info);
	});

	info_window_cluster = new google.maps.InfoWindow;

	return polygon;
	
}


function showArrays(event, cluster_info) {
	// Since this polygon has only one path, we can call getPath() to return the
	// MVCArray of LatLngs.

	var contentString = '<b>Cluster</b><br>'
		// + 'Clicked location: <br>' + event.latLng.lat() + ','
		// + event.latLng.lng() + '<br>' 
		+ '# of pictures at ' + cluster_info['hour']
		+ ': ' + cluster_info['num_pics'];

	// Replace the info window's content and position.
	info_window_cluster.setContent(contentString);
	info_window_cluster.setPosition(event.latLng);

	info_window_cluster.open(map);
}


function linspace(a,b,n) {
	if(typeof n === "undefined") n = Math.max(Math.round(b-a)+1,1);
	if(n<2) { return n===1?[a]:[]; }
	var i,ret = Array(n);
	n--;
	for(i=n;i>=0;i--) { ret[i] = (i*b+(n-i)*a)/n; }
	return ret;
}
