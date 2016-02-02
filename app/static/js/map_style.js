
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

