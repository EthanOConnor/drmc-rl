package drmarioai;

import static drmarioai.Color.*;
import static drmarioai.Searcher.*;
import static drmarioai.Tile.*;

public class DefaultEvaluator implements Evaluator {

  @Override
  public double evaluate(final int[][] playfield) {
    return 100.0 * measureVirsues(playfield)
        + measureClusters(playfield)
        + measureVirusColors(playfield)       
        + measureTiles(playfield)
        + measureHeights(playfield);        
  }
  
  private double measureTiles(final int[][] playfield) {
    int total = 0;
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        if (playfield[y][x] != 0) {
          total++;
        }
      }
    }
    return (128 - total) / 128.0;
  }
  
  private double measureVirsues(final int[][] playfield) {
    int total = 0;
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        if ((playfield[y][x] & TILE_MASK) == VIRUS) {
          total++;
        }
      }
    }
    if (total == 0) {
      return 1.0;
    }
    return (128 - total) / 128.0;
  }
  
  private double measureVirusColors(final int[][] playfield) {
    int total = 0;
    int viruses = 0;
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        if ((playfield[y][x] & TILE_MASK) == VIRUS) {
          viruses++;
          final int color = playfield[y][x] & COLOR_MASK;
          for(int i = WIDTH - 1; i >= 0; i--) {
            final int c = playfield[y][i] & COLOR_MASK;
            if (i != x) {
              if (c == color) {
                final int d = x - i;
                total += (225 - d * d) << 3;
              } else if (c == BLACK) {
                final int d = x - i;
                total += 225 - d * d;
              }
            }
          }
          for(int i = HEIGHT - 1; i >= 0; i--) {
            final int c = playfield[i][x] & COLOR_MASK;
            if (i != y) {
              if (c == color) {
                final int d = y - i;
                total += (225 - d * d) << 3;
              } else if (c == BLACK) {
                final int d = y - i;
                total += 225 - d * d;
              }
            }
          }
        }
      }
    }
    if (viruses == 0) {
      return 1.0;
    }
    
    return total / (19840.0 * viruses);
  }
  
  private double measureClusters(final int[][] playfield) {
    int changes = 0;
    for(int y = HEIGHT - 1; y >= 0; y--) {
      int lastColor = BLACK;
      for(int x = WIDTH - 1; x >= 0; x--) {
        final int color = playfield[y][x] & COLOR_MASK;
        if (color != BLACK && color != lastColor) {
          changes++;
          lastColor = color;
        }
      }
    }
    
    for(int x = WIDTH - 1; x >= 0; x--) {
      int lastColor = BLACK;
      for(int y = HEIGHT - 1; y >= 0; y--) {
        final int color = playfield[y][x] & COLOR_MASK;
        if (color != BLACK && color != lastColor) {
          changes++;
          lastColor = color;
        }
      }
    }
    
    return (256 - changes) / 256.0;
  }
  
  private double measureHeights(final int[][] playfield) {
    int total = 0;
    for(int x = WIDTH - 1; x >= 0; x--) {
      for(int y = 0; y < HEIGHT; y++) {
        if (playfield[y][x] != 0) {
          if (y != 0) {
            total += 16.0;
          }
          total += y;
          break;
        }
      }
    }
    return total / 256.0;
  }
}
