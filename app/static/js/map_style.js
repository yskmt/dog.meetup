
var	main_color = '#0085a1',
	saturation_value= -20,
	brightness_value= 5;
var styles = [ 
	{
		elementType: "labels",
		stylers: [
			{saturation: saturation_value}
		]
	},  
	{
		featureType: "poi",
		elementType: "labels",
		stylers: [
			{visibility: "off"}
		]
	},
	{
		featureType: 'road.highway',
		elementType: 'labels',
		stylers: [
			{visibility: "off"}
		]
	}, 
	{
		featureType: "road.local", 
		elementType: "labels.icon", 
		stylers: [
			{visibility: "off"} 
		] 
	},
	{
		featureType: "road.arterial", 
		elementType: "labels.icon", 
		stylers: [
			{visibility: "off"}
		] 
	},
	{
		featureType: "road",
		elementType: "geometry.stroke",
		stylers: [
			{visibility: "off"}
		]
	},
	{ 
		featureType: "transit", 
		elementType: "geometry.fill", 
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	}, 
	{
		featureType: "poi",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "poi.government",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "poi.sport_complex",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "poi.attraction",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "poi.business",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "transit",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "transit.station",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "landscape",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
		
	},
	{
		featureType: "road",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	},
	{
		featureType: "road.highway",
		elementType: "geometry.fill",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	}, 
	{
		featureType: "water",
		elementType: "geometry",
		stylers: [
			{ hue: main_color },
			{ visibility: "on" }, 
			{ lightness: brightness_value }, 
			{ saturation: saturation_value }
		]
	}
];


// var styles = [ 
// 	{featureType: 'poi.attraction', 
// 	 stylers: [{color: '#fce8b2'}] 
// 	}, 
// 	{featureType: 'road.highway', 
// 	 stylers: [{hue: '#0277bd'}, {saturation: -50}] 
// 	}, 
// 	{featureType: 'road.highway', 
// 	 elementType: 'labels.icon', 
// 	 stylers: [{hue: '#000'}, {saturation: 100}, {lightness: 50}] 
// 	}, 
// 	{featureType: 'landscape', 
// 	 stylers: [{hue: '#259b24'}, {saturation: 10}, {lightness: -22}] 
// 	} 
// ]; 


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
