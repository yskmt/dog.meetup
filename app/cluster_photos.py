import numpy as np
from numpy import cos, sin, arccos, arctan2

import pandas as pd
from sklearn.cluster import KMeans


# y_pred = KMeans(n_clusters=20, random_state=0).fit_predict(photo_data_from_sql[['latitude', 'longitude']])

# photo_data_from_sql['label'] = y_pred
# photo_data_from_sql.to_csv('results_labeled.csv', encoding='utf-8')


def latlon_to_dist(latlon, center, R=6371.0):
    """
    Map latitude + longitude cooridnates to km coordinates based
    on reference point.
    
    NOTE: latlon, center are in DEGREES.

    """

    latlon = np.deg2rad(latlon)
    center = np.deg2rad(center)
    
    # center latitude, longitude
    phi_1, lam_0 = center

    # latitude and longitude
    phi, lam = latlon

    # stereographic
    # k = 2 * R / (1 + sin(phi_1)*sin(phi) + cos(phi_1)*cos(phi)*cos(lam-lam_0))
    # x = k * cos(phi) * sin(lam - lam_0)
    # y = k * (cos(phi_1)*sin(phi) - sin(phi_1)*cos(phi)*cos(lam-lam_0))

    # Azimuthal Equidistant Projection
    # c = arccos(sin(phi_1) * sin(phi) + cos(phi_1)*cos(phi)*cos(lam-lam_0))
    # k = c / sin(c)
    # x = k * cos(phi) * sin(lam-lam_0)
    # y = k * (cos(phi_1)*sin(phi) - sin(phi_1)*cos(phi)*cos(lam-lam_0))
    
    # dlon = 0
    dlat = latlon[0] - center[0]
    a = sin(dlat/2)**2
    c = 2 * arctan2(np.sqrt(a), np.sqrt(1-a))
    y = R*c

    # dlat = 0
    dlon = latlon[1] - center[1]
    a = cos(center[0])*cos(latlon[0])*sin(dlon/2)**2
    c = 2 * arctan2(np.sqrt(a), np.sqrt(1-a))
    x = R*c

    # a = sin(dlat/2)**2 + cos(center[0])*cos(latlon[0])*(sin(dlon/2)**2)
    # c = 2 * arctan2(np.sqrt(a), np.sqrt(1-a))
    # d = R*c

    return x, y



print latlon_to_dist((37.751111, -122.433898), (37.760888999999999, -122.446532))
