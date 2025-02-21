import sys
sys.path.append('../')
from utilities import compute_bbox_center, compute_euclidean_distance

class BallAssigner:
    def __init__(self):
        # threshold to decide if the ball is close enough to a player
        self.player_ball_threshold = 70

    def find_ball_holder(self, player_dict, ball_box):
        # get the center of the ball
        ball_center = compute_bbox_center(ball_box)

        # track closest distance and chosen player
        closest_distance = float('inf')
        chosen_player = -1

        # check each player to find who is closest to the ball
        for pid, pdata in player_dict.items():
            pbox = pdata['bbox']

            # measure distance from left foot and right foot
            left_dist = compute_euclidean_distance((pbox[0], pbox[3]), ball_center)
            right_dist = compute_euclidean_distance((pbox[2], pbox[3]), ball_center)
            current_dist = min(left_dist, right_dist)

            # update chosen player if this distance is closer
            if current_dist < self.player_ball_threshold:
                if current_dist < closest_distance:
                    closest_distance = current_dist
                    chosen_player = pid

        return chosen_player
