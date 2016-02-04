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


function draw_dog_marker(i, d, pinColors, map, icon_url,
						 directionsDisplays, directionsService, k){
	
	var pc = pinColors(parseFloat(d["label"])).slice(1);
	var latlng = {lat: parseFloat(d["latitude"]),
				  lng: parseFloat(d["longitude"])};
	
	var pinColor = pc;
	var pinImage = { 
		url: icon_url,
		scaledSize: new google.maps.Size(50, 50),
		origin: new google.maps.Point(0, 0), // origin
		anchor: new google.maps.Point(25, 25) // anchor
	};

	var circle = new google.maps.Marker({
		position: latlng,
		map: map,
		title: 'top',
		icon: pinImage,
		zIndex: 1000000,
		optimized: false
	});
	
	var contentstring = "<p>" + d["kde_score_2"] + "</p>";
	var info_window = new google.maps.InfoWindow();
	
	circle.addListener('click', function(event) {

		open_dog_info(event.latLng.lat(), event.latLng.lng(),
					  info_window);

	}); /* addListner */

	directionsDisplays.push(
		new google.maps.DirectionsRenderer({
			polylineOptions: {
				strokeColor: '#000000'
			},
			preserveViewport: true
		})
	);
	
	calculateAndDisplayRoute(directionsService, directionsDisplays[k],
							 myLatLng, latlng);
	directionsDisplays[k].setMap(map);
	k += 1;
	
	circle.setMap(map);
	
	return {info_window: info_window,
			marker: circle,
			k: k};
}


function open_dog_info(lat, lng, info_window){
	
	/* ajax request */
	$.getJSON('/_add_numbers', {
		lat: lat,
		lon: lng,
		lat_c: myLatLng.lat,
		lon_c: myLatLng.lng,
		kde_score_max: kde_score_max,
		tempfile: tempfile
	}, function(data) {

		score = [];
		$.each(data.result[0], function(i,d){
			/* v: time of day, f: string */
			if (+i == +hour_24){
				score.push([{v: [(+i), 0, 0], f: i}, d, 'red', getAMPM(+i)]);
			}
			else{
				score.push([{v: [(+i), 0, 0], f: i}, d, 'color: #76A7FA', getAMPM(+i)]);
			}
		});

		var data = new google.visualization.DataTable();
		data.addColumn('timeofday', 'Time of a Day');
		data.addColumn('number', '');
		data.addColumn({ type: 'string', role: 'style' });
		data.addColumn({ type: 'string', role: 'tooltip' });
		
		data.addRows(score);
		
		var options = {
			title: 'Dog Level Throughout a Day (out of 5)',
			hAxis: {
				title: 'Time of a Day',
				format: 'h a',
				viewWindow: {
					min: [0, 0, 0],
					max: [24, 0, 0]
				},
				ticks: [[0,0,0], [6,0,0], [12,0,0], [18,0,0], [24,0,0]]
			},
			vAxis: {
				title: 'level',
				minValue: 0,
				maxValue: 5.5,
				ticks: [0, 1, 2, 3, 4, 5]
			},
			legend: 'none'
		};

		var node = document.createElement('div'),
			chart = new google.visualization.ColumnChart(node);
		
		chart.draw(data, options);
		
		/* Replace the info window's content and position. */
		info_window.setContent(node);
		info_window.setPosition({lat: lat, lng: lng});
		info_window.open(map);
	});


}

function shuffle(o){
    for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
    return o;
}
