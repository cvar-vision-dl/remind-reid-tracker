class Detection:
    """
    Detección puntual en un frame.
    """

    def __init__(
        self,
        detection_id,
        class_id,
        frame_id,
        timestamp,
        bbox=None,
        mask=None,
        image_ref=None,
        confidence: float = 0.0,
        geom: dict | None = None,
    ):
        self.detection_id = detection_id
        self.frame_id = frame_id
        self.timestamp = timestamp

        self.class_id = class_id
        self.bbox = bbox
        self.mask = mask
        self.confidence = confidence

        self.geom = geom

        self.crop_img = None
        self.crop_mask = None
        self.crop_bbox = None
        self.crop_transform = None
        self.bucket_transform = None

    def set_crop_data(self, crop_img, crop_mask, crop_bbox, crop_transform, bucket_transform):
        self.crop_img = crop_img
        self.crop_mask = crop_mask
        self.crop_bbox = crop_bbox
        self.crop_transform = crop_transform
        self.bucket_transform = bucket_transform