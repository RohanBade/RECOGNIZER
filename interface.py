import numpy as np
import pygame as pg
from PIL import Image
import pandas as pd
from nn import neuralNetwork

class UI:
    def __init__( self ):
        self.neural_net = neuralNetwork( 784, 100, 10, 0.3 )
        self.neural_net.train_csv( "assets/mnist_train_100.csv", 10 )

        pg.init()
        pg.display.set_caption( "Number Prediction Using NN" )

        self.running = True
        
        self.screen_wh = ( 400, 600 )
        self.draw_wh = ( 400, 400 )
        self.draw_xy = ( 0, 0 )    
    
        self.display_wh = ( 30, 30 )

        self.button_wh = ( 200, 50 )
        
        self.clear_xy = ( 20, 450 )
        self.predict_xy = ( 200, 450 )

        self.prediction = ( 20, 540 )
        self.brush_radius = 10

        self.screen = pg.display.set_mode( self.screen_wh )
        
        self.start_draw = False
        self.end_draw = False

        self.draw_surface = pg.Surface( self.draw_wh )
        self.screen.fill( pg.Color( 144, 149, 162 ) )

        self.draw_surface.fill( "white" )

        clear = pg.image.load( "assets/clear.png" )
        self.clear = pg.transform.scale( clear, self.button_wh )
        self.clear_b = self.clear.get_rect()
        self.clear_b.x, self.clear_b.y = self.clear_xy
        
        predict = pg.image.load( "assets/predict.png" )
        self.predict = pg.transform.scale( predict, self.button_wh )
        self.predict_b = self.predict.get_rect()
        self.predict_b.x, self.predict_b.y = self.predict_xy

        self.font = pg.font.SysFont( "Consolas", 30 )
        self.surf_f = self.font.render( "Predicted number: ", False, ( 0, 0, 0 ) )
        self.screen.blit( self.surf_f, self.prediction )

        self.screen.blit( self.clear, self.clear_b)
        self.screen.blit( self.predict, self.predict_b)
        self.screen.blit( self.draw_surface, self.draw_xy )

        pg.draw.rect( self.screen, pg.Color( 255, 255, 255 ), pg.Rect( 330 ,525, 40, 50 ) )

    def inside( self, tup ):
        if tup[0] > ( self.draw_xy[0] + self.brush_radius ) and tup[0] < ( self.draw_wh[0] + self.draw_xy[0] - self.brush_radius  ) and tup[1] > ( self.draw_xy[1] + self.brush_radius ) and tup[1] < ( self.draw_wh[1] + self.draw_xy[1] - self.brush_radius  ) :
            return True
        return False

    def clear( self ):
        self.draw_surface.fill( "white" )
    
    def image_to_file( self, img ):
        image = Image.open( img )
        resized_image = image.resize( ( 28, 28 ) )
        resized_image.save( "images/new_letter.png" )
        img_gray = resized_image.convert( 'L' )
        arr = np.array( img_gray ).flatten()
        label = self.neural_net.test_csv( arr )

        self.font = pg.font.SysFont( "Consolas", 40 )
        self.display_value = self.font.render( f"{str( label )}", False, ( 0, 0, 0 ) )
        pg.draw.rect( self.screen, pg.Color( 255, 255, 255 ), pg.Rect( 330 ,525, 40, 50 ) )
        self.screen.blit( self.display_value, ( 335, 535 ) )

    def run( self ):

        while self.running :
            
            for event in pg.event.get():

                if event.type == pg.QUIT :
                    self.running = False

                if event.type == pg.KEYDOWN :
                    if event.key == pg.K_ESCAPE :
                        self.running = False

                if event.type == pg.MOUSEBUTTONDOWN :
                    self.start_draw = True
                    self.end_draw = False
                    
                    ( x, y ) = pg.mouse.get_pos()

                    collision_clear = self.clear_b.collidepoint( ( x, y ) )
                    if collision_clear :
                        self.draw_surface.fill( "white" )
                        self.screen.blit( self.draw_surface, self.draw_xy )
                    
                    collision_predict = self.predict_b.collidepoint( ( x, y ) )
                    if collision_predict :
                        pg.image.save( self.draw_surface, "images/letter.png" )
                        self.image_to_file( "images/letter.png" )
                
                if event.type == pg.MOUSEBUTTONUP :
                    self.start_draw = False
                    self.end_draw = True        

            if self.start_draw == True and self.end_draw == False :
                
                ( x, y ) = pg.mouse.get_pos()
                if self.inside( ( x, y ) ) :
                    pg.draw.circle( self.draw_surface, "black", ( x, y ), self.brush_radius )
                    pg.draw.circle( self.screen, "black", ( x, y ), self.brush_radius )
            pg.display.flip()        
            
        pg.quit()
