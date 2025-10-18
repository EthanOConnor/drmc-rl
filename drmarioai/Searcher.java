package drmarioai;

import static drmarioai.Color.*;
import static drmarioai.Move.*;
import static drmarioai.Orientation.*;
import static drmarioai.Tile.*;

public class Searcher {
  
  public static final int[][] ORIENTATIONS = {
    { NoMove, Rotate90, Rotate180, Rotate_90, },
    { Rotate_90, NoMove, Rotate90, Rotate180, },
    { Rotate180, Rotate_90, NoMove, Rotate90 },
    { Rotate90, Rotate180, Rotate_90, NoMove, },
  };
  
  public static final int WIDTH = 8;
  public static final int HEIGHT = 16;  
  
  private int[][] playfield = new int[HEIGHT][WIDTH];
  private boolean[][] removed = new boolean[HEIGHT][WIDTH];
  private int[][][] moves = new int[HEIGHT][WIDTH][4];
  private Pill[] queue = new Pill[WIDTH * HEIGHT * 32];
  private int queueHead;
  private int queueTail;
  private TargetListener targetListener;
  
  public Searcher() {
    for(int i = queue.length - 1; i >= 0; i--) {
      queue[i] = new Pill();
    }
  }

  public void setTargetListener(final TargetListener targetListener) {
    this.targetListener = targetListener;
  }
  
  public void setPlayfield(final int[][] playfield) {
    for(int i = HEIGHT - 1; i >= 0; i--) {
      System.arraycopy(playfield[i], 0, this.playfield[i], 0, WIDTH);
    }   
  }
  
  public int[][] getPlayfield() {
    return playfield;
  }
  
  public int getMoves(int x, int y, int orientation, final int[] list) {    
    int i = -1;
    while(true) {
      final int m = moves[y][x][orientation];
      final int move = m;
      if ((move & 0xFF) == Spawn) {
        break;
      }
      list[++i] = move;      
      x = (m >> 8) & 0xFF;
      y = (m >> 16) & 0xFF;
      orientation = (m >> 24) & 0xFF;
    }
    return i;
  }
  
  public void lockPill(final Pill pill, final int color1, final int color2) {
    lockPill(pill.x, pill.y, pill.orientation, color1, color2);
  }
  
  public void lockPill(final int x, final int y, 
      final int orientation, final int color1, final int color2) {
    
    switch(orientation) {
      case HORIZONTAL:
        lockTile(x, y, color1, LEFT);
        lockTile(x + 1, y, color2, RIGHT);
        break;
      case VERTICAL:
        lockTile(x, y, color1, BOTTOM);
        lockTile(x, y - 1, color2, TOP);
        break;
      case REVERSED_HORIZONTAL:
        lockTile(x, y, color2, LEFT);
        lockTile(x + 1, y, color1, RIGHT);
        break;
      case REVERSED_VERTICAL:
        lockTile(x, y, color2, BOTTOM);
        lockTile(x, y - 1, color1, TOP);        
        break;
    }
    
    while(removeConnections()) {
      dropUnsupported();
    }
  }
  
  private void lockTile(final int x, final int y, final int color, 
      final int tile) {
    
    if (x >= 0 && y >= 0 && x < WIDTH && y < HEIGHT) {
      if (y == 0 && tile == BOTTOM) {
        playfield[y][x] = color | SQUARE;
      } else {
        playfield[y][x] = color | tile;
      }
    }
  }
  
  private void dropUnsupported() {
    for(int y = HEIGHT - 2; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        switch(playfield[y][x] & TILE_MASK) {
          case SQUARE:
            if (playfield[y + 1][x] == 0) {
              dropSquare(x, y);
            }
            break;
          case BOTTOM:
            if (playfield[y + 1][x] == 0) {
              dropVerticalPill(x, y);
            }
            break;
          case RIGHT:
            if (playfield[y + 1][x] == 0 && playfield[y + 1][x - 1] == 0) {
              dropHorizontalPill(x, y);
            }
            break;
        }
      }
    }
  }
  
  private void dropHorizontalPill(final int x, final int y) {
    int newY1 = y;
    while(newY1 != HEIGHT - 1 && playfield[newY1 + 1][x] == 0) {
      newY1++;
    }
    
    int newY2 = y;
    while(newY2 != HEIGHT - 1 && playfield[newY2 + 1][x - 1] == 0) {
      newY2++;
    }
    
    int newY = Math.min(newY1, newY2);
    playfield[newY][x] = playfield[y][x];
    playfield[y][x] = 0;
    playfield[newY][x - 1] = playfield[y][x - 1];
    playfield[y][x - 1] = 0;
  }
  
  private void dropVerticalPill(final int x, final int y) {
    int newY = y;
    while(newY != HEIGHT - 1 && playfield[newY + 1][x] == 0) {
      newY++;
    }
    playfield[newY][x] = playfield[y][x];
    playfield[y][x] = 0;
    playfield[newY - 1][x] = playfield[y - 1][x];
    playfield[y - 1][x] = 0;
  }
  
  private void dropSquare(final int x, final int y) {
    int newY = y;
    while(newY != HEIGHT - 1 && playfield[newY + 1][x] == 0) {
      newY++;
    }
    playfield[newY][x] = playfield[y][x];
    playfield[y][x] = 0;
  }
  
  private void turnHalvesToSquares() {
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        switch(playfield[y][x] & TILE_MASK) {
          case LEFT:
            if ((playfield[y][x + 1] & TILE_MASK) != RIGHT) {
              turnHalfToSquare(x, y);
            }
            break;
          case RIGHT:
            if ((playfield[y][x - 1] & TILE_MASK) != LEFT) {
              turnHalfToSquare(x, y);
            }
            break;
          case TOP:
            if ((playfield[y + 1][x] & TILE_MASK) != BOTTOM) {
              turnHalfToSquare(x, y);
            }
            break;
          case BOTTOM:
            if ((playfield[y - 1][x] & TILE_MASK) != TOP) {
              turnHalfToSquare(x, y);
            }
            break;
        }
      }
    }
  }
  
  private void turnHalfToSquare(final int x, final int y) {
    playfield[y][x] = (playfield[y][x] & COLOR_MASK) | SQUARE;
  }
  
  private boolean removeConnections() {
    
    boolean result = false;
    
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        removed[y][x] = false;
      }
    }
    
    for(int y = HEIGHT - 1; y >= 0; y--) {
      int lastColor = BLACK;
      int lastX = WIDTH;
      for(int x = WIDTH - 1; x >= -1; x--) {
        final int color = (x == -1) ? BLACK : (playfield[y][x] & COLOR_MASK);
        if (color != lastColor) {
          final int length = lastX - x;
          if (length >= 4 && lastColor != BLACK) {
            result = true;
            for(int i = length; i > 0; i--) {
              removed[y][x + i] = true;
            }
          }
          lastColor = color;
          lastX = x;
        }
      }
    }
    
    for(int x = WIDTH - 1; x >= 0; x--) {
      int lastColor = BLACK;
      int lastY = HEIGHT;
      for(int y = HEIGHT - 1; y >= -1; y--) {
        final int color = (y == -1) ? BLACK : (playfield[y][x] & COLOR_MASK);
        if (color != lastColor) {
          final int length = lastY - y;
          if (length >= 4 && lastColor != BLACK) {
            result = true;
            for(int i = length; i > 0; i--) {
              removed[y + i][x] = true;
            }
          }
          lastColor = color;
          lastY = y;
        }
      }
    }
    
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        if (removed[y][x]) {
          playfield[y][x] = 0;
        }
      }
    }
    
    turnHalvesToSquares();
    
    return result;
  }
  
  public boolean canSpawn() {
    return playfield[0][3] == 0 && playfield[0][4] == 0;
  }
  
  public boolean containsViruses() {
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        if ((playfield[y][x] & TILE_MASK) == VIRUS) {
          return true;
        }
      }
    }
    return false;
  }
  
  public void search() {
    
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        for(int o = 3; o >= 0; o--) {
          moves[y][x][o] = NoMove;
        }  
      }
    }
    
    clearQueue();
    enqueue(3, 0, HORIZONTAL);
    
    while(!isEmpty()) {      
      final Pill p = dequeue();
      if (p.orientation == HORIZONTAL 
          || p.orientation == REVERSED_HORIZONTAL) {
        if (p.y == HEIGHT - 1 || playfield[p.y + 1][p.x] != 0 
            || playfield[p.y + 1][p.x + 1] != 0) {
          foundTarget(p.x, p.y, p.orientation);
        }
      } else {
        if (p.y == HEIGHT - 1 || playfield[p.y + 1][p.x] != 0) {
          foundTarget(p.x, p.y, p.orientation);
        }
      }        
      for(int o = 3; o >= 0; o--) {
        if (o != p.orientation) {
          enqueue(p.x, p.y, o, ORIENTATIONS[p.orientation][o], 
              p.x, p.y, p.orientation);
        }          
      }
      if (p.x != 0) {
        enqueue(p.x - 1, p.y, p.orientation, Left, p.x, p.y, p.orientation);
      }
      if (p.x != WIDTH - 1) {
        enqueue(p.x + 1, p.y, p.orientation, Right, p.x, p.y, p.orientation);
      }
      if (p.y != HEIGHT - 1) {
        enqueue(p.x, p.y + 1, p.orientation, Down, p.x, p.y, p.orientation);
      }
    }
  }
  
  private void foundTarget(final int x, final int y, final int orientation) {
    if (targetListener != null) {
      targetListener.foundTarget(x, y, orientation);
    }
  }
  
  private boolean isEmpty() {
    return queueHead == queueTail;
  }
  
  private Pill dequeue() {
    return queue[queueTail++];
  }
  
  private void enqueue(final int x, final int y, final int orientation) {
    enqueue(x, y, orientation, Spawn, 0xFF, 0xFF, 0xFFFF);
  }
  
  private void enqueue(int x, final int y, final int orientation,
      final int move, final int fromX, int fromY, int fromOrientation) {
    
    if ((orientation == HORIZONTAL || orientation == REVERSED_HORIZONTAL) 
        && x == WIDTH - 1) {
      x--;
    }
    
    if (moves[y][x][orientation] != -1 || playfield[y][x] != 0) {
      return;
    }
    
    moves[y][x][orientation] = (fromOrientation << 24) | (fromY << 16) 
        | (fromX << 8) | move;
    
    if (orientation == HORIZONTAL || orientation == REVERSED_HORIZONTAL) {
      if (playfield[y][x + 1] != 0) {
        return;
      }
    } else if (y != 0 && playfield[y - 1][x] != 0) {
      return;
    }
    
    final Pill pill = queue[queueHead++];
    pill.x = x;
    pill.y = y;
    pill.orientation = orientation;
  }
  
  private void clearQueue() {
    queueHead = queueTail = 0;
  }
}
