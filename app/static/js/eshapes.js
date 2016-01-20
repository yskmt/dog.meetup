Math.radians = function(degrees) {
  return degrees * Math.PI / 180;
};

function get_ellipse_coords(a, b, x, y, angle, k=10){

	xs = [];
	ys = [];

	// beta = -angle*Math.PI/180.0;
	beta = -angle;
	sin_beta = Math.sin(beta);
	cos_beta = Math.cos(beta);
	
	for(var theta=0; theta < (361); theta+=k){
		
		alpha = Math.radians(theta);
		sin_alpha = Math.sin(alpha);
		cos_alpha = Math.cos(alpha);

		xs.push(x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta));
		ys.push(y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta));
	}

	return [xs, ys];
}


function show_polygon(map, xs, ys){

	// Define the LatLng coordinates for the polygon.
	var triangleCoords = [];

	for(var i=0; i<xs.length; i+=1){
		triangleCoords.push({lat: ys[i], lng: xs[i]});
	}

	// Construct the polygon.
	var polygon = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.8,
		strokeWeight: 3,
		fillColor: '#FF0000',
		fillOpacity: 0.35
	});
	polygon.setMap(map);

	return polygon;
	
	// Add a listener for the click event.
	// bermudaTriangle.addListener('click', showArrays);

	// infoWindow = new google.maps.InfoWindow;
}
