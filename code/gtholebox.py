import numpy as np
from typing import List



class GTHoleBox:
    def __init__(self, hole_box: List[float], label: int):
        """

        Parameters:
        ----------
        hole_box : List[float]
            A list of four float values representing the coordinates [x1, y1, x2, y2] of the hole box.
        label : int
            An integer label associated with the hole box.

        Raises:
        ------
        ValueError
            If the `hole_box` list does not contain exactly four elements.
        """        
        if len(hole_box) != 4:
            raise ValueError("hole_box must contain exactly four elements: [x1, y1, x2, y2]")
        
        self.hole_box = hole_box
        self.label = label

    def get_hole_box(self) -> List[float]:
        """
        Returns the coordinates of the hole box.

        Returns:
        -------
        List[float]
            A list of four float values representing the coordinates [x1, y1, x2, y2] of the hole box.
        """        
        return self.hole_box
    
    def get_hole_box_reshape(self):
        """
        Returns the hole box coordinates reshaped into a numpy array format suitable for specific operations.
        
        The reshaped array will represent the four corners of the hole box as a 4x1x2 array.
        
        [x1, y1, x2, y2] = [10, 20, 100, 120]

        (10,20) -------- (100,20)
        (x1,y1) -------- (x2,y1)
            |                |
            |                |
            |                |
        (x1,y2) -------- (x2,y2)
        (10,120)--------(100,120)

        Converting x1, y1, x2, y2 into the following format:
            [
                [10, 20],
                [100, 20],
                [100,120],
                [10,120]
                            ]

        Returns:
        -------
        np.ndarray
            A numpy array with shape (4, 1, 2) representing the four corners of the hole box.
        """        
        arr = np.array([[self.hole_box[0], self.hole_box[1]],  # Top-left
                   [self.hole_box[2], self.hole_box[1]],  # Top-right
                   [self.hole_box[2], self.hole_box[3]],  # Bottom-right
                   [self.hole_box[0], self.hole_box[3]]
                   ], dtype=np.float32)  # Bottom-left        
        return arr.reshape(-1, 1, 2)
    
    def get_hole_id(self) -> int:
        return self.label
    
    def get_box_center(self) -> np.ndarray:
        """
        Returns the center point of the hole box.

        This is computed as the average of the x and y coordinates.

        Returns:
        -------
        np.ndarray
            A numpy array with the center coordinates [center_x, center_y] of the hole box.

        Raises:
        ------
        AssertionError
            If `hole_box` does not contain exactly four elements.
        """        
        assert len(self.hole_box) == 4
        return np.array([
            (self.hole_box[0] + self.hole_box[2]) / 2,
            (self.hole_box[1] + self.hole_box[3]) / 2,
        ])