import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


'''
GameOfLife:

init():
    board: Use a premade board of ones and zeroes. Nullifies board_size and border_size. Note: board has to a square and of type numpy array.
    board_size: The amount of tiles in the visible board.
    border_size: The amount of area outside of the visible board in all directions.

run():
    Returns a touple of the board represented as an array at the start and at endsssssss
    visual: Saves an animation depicting the game of life playing out
    delta: Amount of delta to run the game of life program before returning
'''

class GameOfLife():
    def __init__(self, board=np.array([]), board_size=20, border_size=0):
        if board.size == 0:
            self.__board_size = board_size + border_size * 2
            self.__board = np.array(np.random.randint(2, size=(self.__board_size, self.__board_size)))
            self.__border_size = border_size
        else:
            assert(board.shape[0] == board.shape[1])
            board_size = board.shape[0] + border_size * 2
            self.__board = np.array([[0] * board_size for _ in range(board_size)])
            self.__board_size = board_size
            start = border_size
            end = self.__board_size - start
            self.__board[start:end, start:end] = board.copy()
            self.__border_size = border_size
        self.start = self.__border_size
        self.end = self.__board_size - self.start

    def __count_alive_neighbors(self, x, y):
        alive_neighbors = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0:
                    continue
                neighbor_x = x + i
                neighbor_y = y + j
                if self.__is_valid_position(neighbor_x, neighbor_y):
                    alive_neighbors += self.__board[neighbor_x][neighbor_y]
        return alive_neighbors

    def __is_valid_position(self, x, y):
        return 0 <= x < self.__board_size and 0 <= y < self.__board_size

    def __get_limited_board(self):
        start = self.__border_size
        end = self.__board_size - start
        return self.__board[start:end, start:end]

    def run(self, visual=False, delta=-1):
        starting_board = self.__get_limited_board().copy()
        if visual:
            fig, ax = plt.subplots()
            plt.axis('off')
            self.__img = ax.imshow(self.__board)
            if delta < 0:
                ani = animation.FuncAnimation(fig, self.__update, interval=1)
            else:
                ani = animation.FuncAnimation(fig, self.__update, frames=delta, interval=1, save_count=delta)
            ani.save('GOL_Simulation.avi', fps=2, writer='ffmpeg', codec='rawvideo')
        else:
            self.__img = 0
            for i in range(delta):
                self.__update(i, display=False)
        
        end_board = self.__get_limited_board().copy()
        return starting_board, end_board
        
    def __update(self, frame, display=True):
        next_board = self.__board.copy()
        for i in range(self.__board_size):
            for j in range(self.__board_size):
                neighbors = self.__count_alive_neighbors(j, i)
                if self.__board[j][i]:
                    if neighbors < 2 or neighbors > 3:
                        next_board[j][i] = 0
                else:
                    if neighbors == 3:
                        next_board[j][i] = 1
        self.__board[:] = next_board[:]
        if display:
            self.__img.set_data(self.__get_limited_board())
            return self.__img,

def main():
    board_i = np.array([
        # glider
        # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # acorn
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    gof = GameOfLife(board=board_i, border_size=20)
    gof.run(visual=True, delta=50)

if __name__ == '__main__':
    main()
