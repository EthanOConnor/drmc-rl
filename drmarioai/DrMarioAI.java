package drmarioai;

import nintaco.api.*;
import static drmarioai.Color.*;
import static drmarioai.Move.*;
import static drmarioai.Searcher.*;
import static drmarioai.Tile.*;

public class DrMarioAI {
  
  private static final int MOVES_DELAY = 6;
  private static final int START_DELAY = 90 * 60;
  
  private final API api = ApiSource.getAPI();
  
  private final int[] moves = new int[256];
  private final int[][] playfield = new int[16][8];
  private final SearchChain searchChain = new SearchChain();
  
  private int spawned;
  private int movesIndex = -1;
  private int movesDelay;
  private int startDelay;
  private int lastPillCount = -1;
  private boolean pressA;
  private boolean lastEnding;
  private boolean fast;
  
  public void launch(final boolean fast) {
    this.fast = fast;
    api.addFrameListener(this::renderFinished);
    api.run();
  }
  
  private void renderFinished() {
    
    final int mode = api.readCPU(Address.MODE);
    if (mode != 0x04) {
      lastPillCount = -1;
      return;
    }
    
    final int players = api.readCPU(Address.NUMBER_OF_PLAYERS);
    final int playerAddress = players == 1 ? 0x0000 : 0x0080;
    
    if (players == 1 && api.readCPU(Address.STAGE_CLEARED) == 0x01) {
      final boolean ending = api.readCPU(Address.ENDING) != 0x0A;
      if (!lastEnding && ending) {
        startDelay = START_DELAY;
      }
      lastEnding = ending;
      if (startDelay > 0) {
        startDelay--;
      } else {
        api.writeGamepad(players - 1, GamepadButtons.Start, true);
      }
    }
    
    if (movesIndex >= 0) {
      stallDrop();
      if (movesDelay > 0 && --movesDelay == 0) {
        movesDelay = MOVES_DELAY;
        if (pressA) {
          pressA = false;
          api.writeGamepad(players - 1, GamepadButtons.A, true);
        } else {   
          
          final int x = api.readCPU(playerAddress | Address.CURRENT_X);
          final int y = 15 - api.readCPU(playerAddress | Address.CURRENT_Y);
          final int orientation = api.readCPU(playerAddress 
              | Address.CURRENT_ORIENTATION);

          final int move = moves[movesIndex];
          final int expectedX = (move >> 8) & 0xFF;
          final int expectedY = (move >> 16) & 0xFF;
          final int expectedOrientation = (move >> 24) & 0xFF;
          
          if (x != expectedX || y != expectedY 
                || orientation != expectedOrientation) {
            api.writeCPU(playerAddress | Address.CURRENT_X, expectedX);
            api.writeCPU(playerAddress | Address.CURRENT_Y, 15 - expectedY);
            api.writeCPU(playerAddress | Address.CURRENT_ORIENTATION, 
                expectedOrientation);
          }          
          
          switch(moves[movesIndex--] & 0xFF) {
            case Left:
              api.writeGamepad(players - 1, GamepadButtons.Left, true);
              break;
            case Right:
              api.writeGamepad(players - 1, GamepadButtons.Right, true);
              break;
            case Down:
              forceDrop();
              api.writeGamepad(players - 1, GamepadButtons.Down, true);
              break;
            case Rotate90:
              api.writeGamepad(players - 1, GamepadButtons.B, true);
              break;
            case Rotate180:
              api.writeGamepad(players - 1, GamepadButtons.A, true);
              pressA = true;
              break;
            case Rotate_90:
              api.writeGamepad(players - 1, GamepadButtons.A, true);
              break;
          }
        }
      }
    }
    
    if (spawned > 0 && --spawned == 0) {
      pillSpawned(players, playerAddress);
    }  
      
    final int pillCount = api.readCPU(playerAddress | Address.PILL_COUNT);
    if ((lastPillCount == -1 && pillCount > 1) 
        || (lastPillCount != -1 && lastPillCount != pillCount)) {
      spawned = 3;
    }
    lastPillCount = pillCount;
  }  
  
  private void pillSpawned(final int players, final int playerAddress) {
    readPlayfield(players);
    if (searchChain.search(playfield, 
        readColor(playerAddress | Address.CURRENT_COLOR_1), 
        readColor(playerAddress | Address.CURRENT_COLOR_2), 
        readColor(playerAddress | Address.NEXT_COLOR_1), 
        readColor(playerAddress | Address.NEXT_COLOR_2))) {
      if (players == 1 && fast) {
        api.writeCPU(playerAddress | Address.CURRENT_X, searchChain.getX());
        api.writeCPU(playerAddress | Address.CURRENT_Y, 
            15 - searchChain.getY());
        api.writeCPU(playerAddress | Address.CURRENT_ORIENTATION, 
            searchChain.getOrientation());         
      } else {
        movesIndex = searchChain.getMoves(moves);
        movesDelay = 1;
      }      
    }
  }
  
  private int readColor(final int address) {
    switch(api.readCPU(address) & 0x03) {
      case 1:
        return RED;
      case 2:
        return BLUE;
      default:
        return YELLOW;
    }
  }
  
  private void readPlayfield(final int players) {
    for(int y = HEIGHT - 1; y >= 0; y--) {
      for(int x = WIDTH - 1; x >= 0; x--) {
        final int value = api.readCPU((players == 1 ? Address.P1_PLAYFIELD 
            : Address.P2_PLAYFIELD) | (y << 3) | x);
        final int color;
        switch(value & 0x03) {          
          case 1:
            color = RED;
            break;
          case 2:
            color = BLUE;
            break;
          default:
            color = YELLOW;
            break;
        }
        final int tile;
        switch(value >> 4) {
          case 0x4:
            tile = TOP;
            break;
          case 0x5:
            tile = BOTTOM;
            break;
          case 0x6:
            tile = LEFT;
            break;
          case 0x7:
            tile = RIGHT;
            break;
          case 0x8:
            tile = SQUARE;
            break;
          case 0xB:
          case 0xD:
            tile = VIRUS;
            break;
          default:
            tile = EMPTY;
            break;
        }
        if (tile == EMPTY) {
          playfield[y][x] = 0;
        } else {
          playfield[y][x] = color | tile;
        }
      }
    }
  }
  
  private void stallDrop() {
    api.writeCPU(Address.FRAMES_UNTIL_DROP, 0xFF);
  }
  
  private void forceDrop() {
    api.writeCPU(Address.FRAMES_UNTIL_DROP, 0x01);
  }
  
  public static void main(final String... args) throws Throwable {
    ApiSource.initRemoteAPI("localhost", 9999);
    new DrMarioAI().launch(args.length == 1 
        && "fast".equalsIgnoreCase(args[0]));
  }
}
