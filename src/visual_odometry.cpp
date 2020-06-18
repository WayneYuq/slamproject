VisualOdometry::VisualOdometry() :
  state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), 
  num_lost_ ( 0 ), num_inliers_ ( 0 ) 
{
  num_of_features_ = Config::get<int> ( "number_of_features" );
  scale_factor_ = Config::get<double> ( "scale_factor" );
  level_pyramid_ = Config::get<int> ( "level_pyramid" );
  match_ratio_ = Config::get<float> ( "match_ratio" );
}


bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
  switch ( state_ )
  {
    case: INITIALIZING:
    {
      state_ = OK;
      curr_ = ref_ = frame;
      map_->insertKeyFrame ( frame );
      // extract features from first frame
      extractKeyPoints();\
      computeDescriptors();
      // compute the 3d position of features in ref frame
      setRef3DPoints();
      break;
    }
    case OK:
    {
      curr_ = frame;
      extractKeyPoints();
      computeDescriptors();
      featureMatching();
      poseEstimationPnP();
      
      if ( checkEstimatedPose() == true ) // a good estimation
      {
	curr_->T_c_w_ = T_c_r_estimated_ * ref_-> T_c_w_; // T_c_w_ = T_c_r * T_r_w
	ref_ = curr_;
	setRef3DPoints();
	num_lost_ = 0;
	if ( checkKeyFrame() == true ) // is a key-frame
	{
	  addKeyFrame();
	}
      }
      else // bad estimation due to varuous reasons
      {
	num_lost_++;
	if ( num_lost_ > max_num_lost_ )
	{
	  state_ = LOST;
	}
	return false;
      }
      break;
    }
    case LOST:
    {
      cout << "vo has losts." << endl;
      break;
    }
  }
  return true;
}