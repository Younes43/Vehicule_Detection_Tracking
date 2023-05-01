import motmetrics as mm
import numpy as np
import argparse

def motMetricsEnhancedCalculator(gtSource, tSource):
  # import required packages

  
  # load ground truth
  gt = np.loadtxt(gtSource, delimiter=',')

  # load tracking output
  t = np.loadtxt(tSource, delimiter=',')

  # Create an accumulator that will be updated during each frame
  acc = mm.MOTAccumulator(auto_id=True)

  # Max frame number maybe different for gt and t files
  for frame in range(int(gt[:,0].max())):
    frame += 1 # detection and frame numbers begin at 1

    # select id, x, y, width, height for current frame
    # required format for distance calculation is X, Y, Width, Height \
    # We already have this format
    gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
    t_dets = t[t[:,0]==frame,1:6] # select all detections in t

    C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=0.7) # format: gt, t
    # if frame in [i for i in range(1,101)]:
    #   # print("Frame : ",frame)
    #   # print(C)

    # Call update once for per frame.
    # format: gt object ids, t object ids, distance
    acc.update(gt_dets[:,0].astype('int').tolist(), \
              t_dets[:,0].astype('int').tolist(), C)

  mh = mm.metrics.create()

  summary = mh.compute(acc, metrics=['num_frames', \
                                     'recall', 'precision', 'num_objects', \
                                      'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                      'mota' \
                                    ], \
                      name='acc')

  strsummary = mm.io.render_summary(
      summary,
      #formatters={'mota' : '{:.2%}'.format},
      namemap={ 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
                'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'mota': 'MOTA'  \
              }
  )
  print(strsummary)

def main(gtSource,tSource):
    motMetricsEnhancedCalculator(gtSource, tSource)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Calulate MOT metrics.")
    parser.add_argument("--gt",
                        help="Ground truth file",
                        default="MVI_39031.csv")
    parser.add_argument("--pred",
                        help="Prediction file",
                        default="MVI_39031_output_filtered.csv")
    args = parser.parse_args()
    gtSource=args.gt
    tSource=args.pred
    main(gtSource,tSource)
