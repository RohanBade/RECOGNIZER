import numpy as np
import pandas as pd
import scipy.special

class neuralNetwork:
  def __init__ ( self, inodes, hnodes, onodes, lr ):

    self.inodes = inodes
    self.hnodes = hnodes
    self.onodes = onodes
    self.lr = lr

    self.wih = np.random.normal( 0, inodes ** ( -0.5 ), ( hnodes, inodes ) )
    self.who = np.random.normal( 0, hnodes ** ( -0.5 ), ( onodes, hnodes ) )

    self.sigmoid = lambda x: scipy.special.expit( x )

  def train ( self, inputs, outputs ):

    in_  = np.array( inputs, ndmin = 2 ).T
    out_ = np.array( outputs, ndmin = 2 ).T

    h_inputs = np.dot( self.wih, in_ )
    h_outputs = self.sigmoid( h_inputs )

    final_inputs = np.dot( self.who, h_outputs )
    final_outputs = self.sigmoid( final_inputs )

    errors = out_ - final_outputs

    h_errors = np.dot( self.who.T, errors )

    self.who += self.lr * np.dot( errors * final_outputs * ( 1 - final_outputs ), h_outputs.T )
    self.wih += self.lr * np.dot( h_errors * h_outputs * ( 1 - h_outputs ), in_.T )

  def query ( self, inputs ):
    in_  = np.array( inputs, ndmin = 2 ).T

    h_inputs = np.dot( self.wih, in_ )
    h_outputs = self.sigmoid( h_inputs )

    final_inputs = np.dot( self.who, h_outputs )
    final_outputs = self.sigmoid( final_inputs )

    return final_outputs

  def train_csv( self, csv_name, epochs ):
    training_file = open( csv_name, "r" )
    training_data = training_file.readlines()
    training_file.close()

    for e in range(epochs):
      for record in training_data:
        values = record.split( "," )
        in_ = np.asfarray( values[1:] ) / 255.0 * 0.99 + 0.01
        targets_ = np.zeros( self.onodes ) + 0.01
        targets_[int( values[0] )] = 0.99
        self.train( in_, targets_ )

  def test_csv( self, arr ):
    in_ = 255.0 - np.asfarray( arr )
    in_ = ( in_ / 255.0 ) * 0.99 + 0.01 
    output = self.query( in_ )
    label = np.argmax( output )
    return label

