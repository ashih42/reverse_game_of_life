import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
TODO:

Import a board of rectangular
Export board after x iterations
Command line arguments
Window should not be opening when visual is set to false

"""


"""

GameOfLife:

init:
    board: Use a premade board of ones and zeroes. Nullifies board_size and border_size. Note: board has to a square and of type numpy array.
    board_size: The amount of tiles in the visible board.
    border_size: The amount of area outside of the visible board in all directions.

run:
    save_anim: Saves the animation as a HTML document with a folder of pictures corresponding to each frame of the animation
    visual: Displays an animation depicting the game of life playing out
    epochs: Amount of epochs to run the game of life program before returning
    print_end_state: Prints the board and the end of the epochs

"""

class GameOfLife():

    def __init__(self, board=np.array([]), board_size=20, border_size=20):
        if board.size == 0:
            self.board_size = board_size + border_size * 2
            self.board = np.array(np.random.randint(2, size=(self.board_size, self.board_size)))
            self.border_size = border_size
        else:
            assert(board.shape[0] == board.shape[1])
            board_size = board.shape[0] + border_size * 2
            self.board = np.array([[0] * board_size for _ in range(board_size)])
            self.board_size = board_size
            start = border_size
            end = self.board_size - start
            self.board[start:end, start:end] = board.copy()
            self.border_size = border_size

    def __neighboring_cells(self, x, y):
        result = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j == 0:
                    if x + i < self.board_size and x + i >= 0 and y + j < self.board_size and y + j >= 0:
                        result += self.board[x + i][y + j]
        return result

    def place_glider(self, x, y):
        self.board[y][x] = 0
        self.board[y][x+1] = 1
        self.board[y][x+2] = 0
        self.board[y+1][x] = 0
        self.board[y+1][x+1] = 0
        self.board[y+1][x+2] = 1
        self.board[y+2][x] = 1
        self.board[y+2][x+1] = 1
        self.board[y+2][x+2] = 1

    def run(self, save_anim=False, visual=True, epochs=-1, print_end_state=False):
        #for i in range(self.board_size):
        #    for j in range(self.board_size):
        #        self.board[j][i] = 0
        
        #self.place_glider(self.board_size // 2, self.board_size // 2)

        if visual:
            fig, ax = plt.subplots()
            plt.axis('off')
            self.img = ax.imshow(self.board)

            if epochs < 0:
                ani = animation.FuncAnimation(fig, self.__update, interval=1)
            else:
                ani = animation.FuncAnimation(fig, self.__update, frames=epochs, interval=1, save_count=epochs)

            if save_anim:
                ani.save('gof.html', fps=30)
            
            plt.show()
        else:
            self.img = 0
            for i in range(epochs):
                self.__update(i, display=False)
        
        if print_end_state:
            start = self.border_size
            end = self.board_size - start
            for i in self.board[start:end, start:end]:
                print(i)
        
    def __update(self, frame, display=True):
        next_board = self.board.copy()
        for i in range(self.board_size):
            for j in range(self.board_size):
                neighbors = self.__neighboring_cells(j, i)
                if self.board[j][i]:
                    if neighbors < 2 or neighbors > 3:
                        next_board[j][i] = 0
                else:
                    if neighbors == 3:
                        next_board[j][i] = 1
        if display:
            start = self.border_size
            end = self.board_size - start
            self.img.set_data(self.board[start:end, start:end])
            self.board[:] = next_board[:]
            return self.img,

def main():
    board_i = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    gof = GameOfLife()
    gof.run(print_end_state=True, visual=True, epochs=-1)

if __name__ == '__main__':
    main()
                
