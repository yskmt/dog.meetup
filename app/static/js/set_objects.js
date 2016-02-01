// Load the Visualization API and the piechart package.
google.load('visualization', '1.0', {'packages':['corechart']});

// Set a callback to run when the Google Visualization API is loaded.
// google.setOnLoadCallback(drawChart);


function getAMPM(time){
	if (time==0){
		return 12 + ' AM';
	}
	else if (time<=11){
		return time + ' AM';
	}
	else if(time==12){
		return time + ' PM';
	}
	else {
		return (time%12) + ' PM';
	}
}


function linspace(a,b,n) {
	if(typeof n === "undefined") n = Math.max(Math.round(b-a)+1,1);
	if(n<2) { return n===1?[a]:[]; }
	var i,ret = Array(n);
	n--;
	for(i=n;i>=0;i--) { ret[i] = (i*b+(n-i)*a)/n; }
	return ret;
}

Math.radians = function(degrees) {
  return degrees * Math.PI / 180;
};

function get_ellipse_coords(a, b, x, y, angle, num_pics, k ){

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


function show_polygon(map, xs, ys, cluster_info, myLatLng,
					  density, color){

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
	polygon.setMap(null);
	
	// Add a listener for the click event.
	polygon.addListener('click', function (event){
		showArrays(event, cluster_info, myLatLng);
	});

	info_window_cluster = new google.maps.InfoWindow;

	return polygon;
	
}


function showArrays(event, cluster_info, myLatLng) {
	// Since this polygon has only one path, we can call getPath() to return the
	// MVCArray of LatLngs.

	var contentString = '<b>Cluster</b><br>'
		// + 'Clicked location: <br>' + event.latLng.lat() + ','
		// + event.latLng.lng() + '<br>' 
		+ '# of pictures at ' + cluster_info['hour']
		+ ': ' + cluster_info['num_pics'];
		+'<div class="chart"></div>'; 


	// ajax request
	$.getJSON('/_add_numbers', {
		lat: event.latLng.lat(),
		lon: event.latLng.lng(),
		lat_c: myLatLng.lat,
		lon_c: myLatLng.lng
	}, function(data) {

		score = [['timeofday', 'number', { role: 'style' }]];
		$.each(data.result[0], function(i,d){
			// v: time of day, f: string
			score.push([{v: [(+i), 0, 0], f: i}, d, 'gold']);
			});

		var data = google.visualization.arrayToDataTable(score);
		// var data = new google.visualization.DataTable();

		console.log(data);
		//<<< data.addColumn('timeofday', 'Time of Day');
		// <<<data.addColumn('number', 'Dog level');
		
		// data.addRows();
		// data.addRows(score);

		var options = {
			title: 'Dog Level Throughout a Day',
			hAxis: {
				title: 'Time of a Day',
				format: 'h:mm a',
				viewWindow: {
					min: [0, 0, 0],
					max: [24, 5, 0]
				}
			},
			vAxis: {
				title: 'dog score'
			}
		};

        var node        = document.createElement('div'),
            infoWindow  = info_window_cluster,
            chart       = new google.visualization.ColumnChart(node);
		
		chart.draw(data, options);
		
		// Replace the info window's content and position.
		info_window_cluster.setContent(node);
		info_window_cluster.setPosition(event.latLng);
		
		// info_window_cluster.setContent(contentString);
		// info_window_cluster.setPosition(event.latLng);

		info_window_cluster.open(map);
	});


}



