class Frame
{
 public:
  typedef std::shared_ptr<Frame> Ptr;
  unsigned long id_; // id of this frame
  double time_stamp_; // when it is recorded
  SE3 T_c_w_; // transform from world to camera
  Camera::Ptr camera_; // Pinhole RGB-D Camera model
  Mat color_, depth_; // color and depth image
 public: // data members
  Frame();
  Frame( long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
  ~Frame();

  // factory function
  static Frame::Ptr createFrame();

  // find the depth in depth map
  double findDepth( const cv::KeyPoint& kp );

  // Get Camera Center
  Vector3d getCamCenter() const;

  // Check if a point in this frame
  bool isInFrame( const Vector3d& pt_world );
  
};
