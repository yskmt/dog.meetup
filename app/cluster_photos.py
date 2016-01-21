import numpy as np
from numpy import cos, sin, arccos, arctan2, sign

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
    y = sign(dlat)*R*c

    # dlat = 0
    dlon = latlon[1] - center[1]
    a = cos(center[0])*cos(latlon[0])*sin(dlon/2)**2
    c = 2 * arctan2(np.sqrt(a), np.sqrt(1-a))
    x = sign(dlon)*R*c

    # a = sin(dlat/2)**2 + cos(center[0])*cos(latlon[0])*(sin(dlon/2)**2)
    # c = 2 * arctan2(np.sqrt(a), np.sqrt(1-a))
    # d = R*c

    return x, y

def get_bbox(center, radius=10.0, R=6371.0):
    """
    Get a bounding box (square) around center (lat, lng)
    
    return: southwest, northeast lat/lng
    """

    r = radius/2.0
    
    # calculate the bounding box
    x1 = center[1] - np.rad2deg(r/R/np.cos(np.deg2rad(center[0])))
    y1 = center[0] - np.rad2deg(r/R)
    x2 = center[1] + np.rad2deg(r/R/np.cos(np.deg2rad(center[0])))
    y2 = center[0] + np.rad2deg(r/R)

    return [x1, y1, x2, y2]




if __name__ == "__main__":
    print latlon_to_dist(
        (37.751111, -122.433898), (37.760888999999999, -122.446532))
