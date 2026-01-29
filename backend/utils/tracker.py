"""
Simple Object Tracker
Based on Euclidean distance centroid tracking
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class Tracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        # Store centroids: {object_id: (x, y)}
        # Store history: {object_id: [_timestamp, (x, y)]}
        self.history = OrderedDict()
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = []
        self.next_object_id += 1
        
        # Reset ID to prevent overflow if running for long time
        if self.next_object_id > 1000000:
            self.next_object_id = 0

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]

    def update(self, rects):
        """
        Update tracker with new bounding box rectangles.
        Returns: Dictionary of (object_id, centroid)
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)

        # If no objects are currently tracked, register all inputs
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match input centroids to existing object centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distances between all pairs
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find smallest value in each row (closest input to each object)
            rows = D.min(axis=1).argsort()
            
            # Find smallest value in each column
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # If distance is too large, don't associate
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update history
                self.history[object_id].append(input_centroids[col])
                if len(self.history[object_id]) > 10:  # Keep last 10 points
                    self.history[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            # Find unused rows (lost objects)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Find unused cols (new objects)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
