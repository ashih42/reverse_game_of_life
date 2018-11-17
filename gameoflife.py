import numpy as np

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
    Returns a touple of the board represented as an array at the start and at endsssssss
    save_anim: Saves the animation as a HTML document with a folder of pictures corresponding to each frame of the animation
    visual: Displays an animation depicting the game of life playing out
    epochs: Amount of epochs to run the game of life program before returning

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
        self.start = self.border_size
        self.end = self.board_size - self.start

    def __neighboring_cells(self, x, y):
        result = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not i == j == 0:
                    if x + i < self.board_size and x + i >= 0 and y + j < self.board_size and y + j >= 0:
                        result += self.board[x + i][y + j]
        return result

    def get_limited_board(self):
        start = self.border_size
        end = self.board_size - start
        return self.board[start:end, start:end]

    def run(self, save_anim=False, visual=False, epochs=-1):
        starting_board = self.get_limited_board().copy()

        if visual:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

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
        
        end_board = self.get_limited_board().copy()

        return starting_board, end_board
        
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
        self.board[:] = next_board[:]
        if display:
            self.img.set_data(self.get_limited_board())
            return self.img,

def main():
    board_i = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    gof.run(visual=True)
    #for i in range(100):
    #    print(i)
    #    start, end = gof.run(visual=False, epochs=i)
    #for i in start:
    #    print(i)
    #print('\n')
    #    for i in end:
    #        print(i)
    #    print('\n')

if __name__ == '__main__':
    main()
                
