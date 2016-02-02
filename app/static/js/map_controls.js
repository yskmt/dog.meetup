function CenterControl(controlDiv, map) {

	/* Set CSS for the control border. */
	var controlUI = document.createElement('div');
	controlUI.setAttribute("id", "toggleButton")
	controlUI.style.backgroundColor = '#fff';
	controlUI.style.border = '2px solid #fff';
	controlUI.style.borderRadius = '3px';
	controlUI.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
	controlUI.style.cursor = 'pointer';
	controlUI.style.marginBottom = '22px';
	controlUI.style.textAlign = 'center';
	controlUI.title = 'Click to recenter the map';
	controlDiv.appendChild(controlUI);

	/* Set CSS for the control interior. */
	var controlText = document.createElement('div');
	controlText.style.color = 'rgb(25,25,25)';
	controlText.style.fontFamily = 'Roboto,Arial,sans-serif';
	controlText.style.fontSize = '16px';
	controlText.style.lineHeight = '38px';
	controlText.style.paddingLeft = '5px';
	controlText.style.paddingRight = '5px';
	controlText.innerHTML = 'Toggle Markers/Clusters/Heatmap';
	controlUI.appendChild(controlText);

	/* Setup the click event listeners: simply set the
	   map to Chicago. */
	controlUI.addEventListener('click', function() {
		toggleVisualizations();
	});
	
}

function TopControl(controlDiv, map) {

	/* Set CSS for the control border. */
	var controlUI = document.createElement('div');
	controlUI.setAttribute("id", "toggleButton")
	controlUI.style.backgroundColor = '#fff';
	controlUI.style.border = '2px solid #fff';
	controlUI.style.borderRadius = '3px';
	controlUI.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
	controlUI.style.cursor = 'pointer';
	controlUI.style.marginBottom = '22px';
	controlUI.style.textAlign = 'center';
	controlUI.title = 'Click to recenter the map';
	controlDiv.appendChild(controlUI);

	/* Set CSS for the control interior. */
	var controlText = document.createElement('div');
	controlText.style.color = 'rgb(25,25,25)';
	controlText.style.fontFamily = 'Roboto,Arial,sans-serif';
	controlText.style.fontSize = '16px';
	controlText.style.lineHeight = '38px';
	controlText.style.paddingLeft = '5px';
	controlText.style.paddingRight = '5px';
	controlText.innerHTML = '<div id="top-control">' + address + '<br>within '
		+ distance + ' miles' + '<br>at ' +
		hour + '</div>';
	controlUI.appendChild(controlText);
	
}



function calculateAndDisplayRoute(directionsService, directionsDisplay,
								  originLatLng, destLatLng){
	directionsService.route({
		origin: originLatLng,
		destination: destLatLng,
		travelMode: google.maps.TravelMode.WALKING
	}, function(response, status) {
		if (status === google.maps.DirectionsStatus.OK) {
			directionsDisplay.setDirections(response);
		} else {
			window.alert('Directions request failed due to ' + status);
		}
	});
	directionsDisplay.setOptions( { suppressMarkers: true } );
}




function toggleVisualizations() {
	/* markers --> clusters */
	if (is_marker==1){
		clearMarkers();
		showClusters();
		is_marker=2;
	}
	/* heatmap --> markers */
	else if (is_marker==0){
		heatmap.setMap(null);
		clearTop3();
		showMarkers();
		showTop3();
		is_marker=1;
	}
	/* clusters --> heatmap */
	else{
		clearClusters();
		heatmap.setMap(map);
		showTop3();
		is_marker = 0;
	}

}

function clearMarkers() {
	for (var i = 0; i < markers.length; i++) {
		markers[i].setMap(null);
	}
}

function showMarkers() {
	for (var i = 0; i < markers.length; i++) {
		markers[i].setMap(map);
	}
}

function showTop3() {
	for (var i = 0; i < top3_markers.length; i++) {
		top3_markers[i].setMap(map);
	}
}

function clearTop3() {
	for (var i = 0; i < top3_markers.length; i++) {
		top3_markers[i].setMap(null);
	}
}


function clearClusters() {
	for (var i = 0; i < ellipses.length; i++) {
		ellipses[i].setMap(null);
	}
}

function showClusters() {
	for (var i = 0; i < ellipses.length; i++) {
		ellipses[i].setMap(map);
	}
}

