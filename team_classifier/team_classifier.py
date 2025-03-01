from sklearn.cluster import KMeans
import torch
class TeamClassifier:
    def __init__(self):
        # stores cluster centers for two teams
        self.team_centers = {}
        # keeps track of player ID to team mappings
        self.player_team_map = {}

    def create_kmeans_model(self, upper_seg):
        # Convert upper_seg to a NumPy array if it's a tensor
        if isinstance(upper_seg, torch.Tensor):
            upper_seg = upper_seg.cpu().numpy()
        
        # Now reshape the image to a 2D array for KMeans
        reshaped_img = upper_seg.reshape(-1, 3)
        
        # Create and return your KMeans model, for example:
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=2, random_state=0)
        model.fit(reshaped_img)  # This should now work without error.
        return model


    def extract_player_color(self, frame, bbox):
        # crop the player from the frame
        cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # take only the top half
        upper_seg = cropped[: cropped.shape[0] // 2, :]

        # build a kmeans model from the top half
        model = self.create_kmeans_model(upper_seg)
        labels = model.labels_

        # reshape labels to the image shape
        clustered = labels.reshape(upper_seg.shape[0], upper_seg.shape[1])

        # find the cluster that corresponds to the player
        corners = [clustered[0, 0], clustered[0, -1], clustered[-1, 0], clustered[-1, -1]]
        background_cluster = max(set(corners), key=corners.count)
        jersey_cluster = 1 - background_cluster

        # extract that cluster’s color
        jersey_colour = model.cluster_centers_[jersey_cluster]
        return jersey_colour

    def define_team_colors(self, frame, player_detections):
        # gather colors for all players
        all_colors = []
        for _, info in player_detections.items():
            bbox = info["bbox"]
            color_val = self.extract_player_color(frame, bbox)
            all_colors.append(color_val)
        
        # cluster player colors into 2 teams
        model = KMeans(n_clusters=2, init="k-means++", n_init=10)
        model.fit(all_colors)
        
        self.model = model
        self.team_centers[1] = model.cluster_centers_[0]
        self.team_centers[2] = model.cluster_centers_[1]
        
    def classify_player_team(self, frame, bbox, player_id):
        # if player ID is already assigned, just return it
        if player_id in self.player_team_map:
            return self.player_team_map[player_id]
        # extract the color for this player's jersey
        jersey_colour = self.extract_player_color(frame, bbox)
        predicted_team = self.model.predict(jersey_colour.reshape(1, -1))[0] + 1
        
        self.player_team_map[player_id] = predicted_team
        return predicted_team
