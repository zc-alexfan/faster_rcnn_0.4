# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.faster_rcnn.vgg_extractor import vgg_extractor
from easydict import EasyDict as edict

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

if __name__ == '__main__':


  metaInfo = edict()
  

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  assert args.dataset == 'vg'
  assert args.net == 'vgg16'

  args.imdb_name = "vg_alldata_singleton"
  args.imdb_name = "vg_alldata_smalltrain"
  args.imdb_name = "vg_alldata_minitrain"
  args.imdb_name = "vg_alldata_minival"

  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  feature_folder = './data/features/' + args.imdb_name + '/' 
  if not os.path.exists(feature_folder):
    os.makedirs(feature_folder)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, False)
  imdb.competition_mode(on=True)

  metaInfo.imdb_name = args.imdb_name
  metaInfo.imdb_classes = imdb.classes
  metaInfo.imdb_image_index = imdb.image_index

  meta_file = feature_folder + args.imdb_name + ".meta"
  with open(meta_file, 'wb') as f:
      pickle.dump(metaInfo, f, pickle.HIGHEST_PROTOCOL)


   


  print('{:d} roidb entries'.format(len(roidb)))


  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  fasterRCNN = vgg_extractor(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  thresh = 0.0 # default value when vis=False

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)

  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  #all_summary = []
  for i in tqdm(range(num_images)):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, image_summary = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      ###### assume: order does not change
      image_summary.info.image_idx = imdb.image_index[i]
      image_summary.info.im_data = vgg_extractor._detach2numpy(im_data)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5] # (x1, y1, x2, y2)
      # bbox_pred is used to compute deltas 
      

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) \
                    * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                """
                ### should compute my own stds?  
                Normalize: (x - mean)/std = y
                Unnormalize: x = y*std + mean
                (Pdb) cfg.TRAIN.BBOX_NORMALIZE_STDS: [0.1, 0.1, 0.2, 0.2]
                cfg.TRAIN.BBOX_NORMALIZE_MEANS [0.0, 0.0, 0.0, 0.0]

                box_detas: the offset from anchors? (dx, dy, dw, dh)
                rois/boxes: the anchor locations? 
                """

                box_deltas = box_deltas.view(-1, 4) \
                    * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          # adjust boxes by deltas; output in (x1, y1, x2, y2)
          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          # avoid boxes go out of image
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      # recall: datainfo = tensor([ 800.0000,  600.0000,    1.6000]))
      # raw image has size: 500 x 375
      # then it is scaled to: 800 x 600, 1.6 is the scale
      # scale boxes back to the original size
      pred_boxes /= data[1][0][2].item()  # (x1, y1, x2, y2)

      # phase 1

      # each proposal has a prob distri.
      scores = scores.squeeze() # torch.Size([300, 151])

      # each proposal has 604 bboxes, one box for each class
      pred_boxes = pred_boxes.squeeze() # torch.Size([300, 604]), 604=4*151


      detect_time = time.time() - det_tic
      misc_tic = time.time()

      # non-maximal suppresion on each class
      # phase 2

      """
      loop over each class, consider each class separately
      i.e. each class has 300 proposals
      at each iteartion, nms, to decide which of the 300 proposals it should keep.
      """
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            # scores[:, j].shape == 300 
            # scores[:, j][inds] just for subset of that 300 boxes
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            """
            (Pdb) cls_boxes.shape
            torch.Size([300, 4])
            (Pdb) cls_scores.shape
            torch.Size([300])
            (Pdb) cls_dets.shape
            torch.Size([300, 5])
            """
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]

            """
            (Pdb) keep
            tensor([[   0], [   3], [  10], [  14], [  27],
                    [  29], [  31], [  45], [  70], [  76],
                    [  83], [ 103], [ 114], [ 115], [ 133],
                    [ 135], [ 136]], dtype=torch.int32, device='cuda:0')
            """
            keep = nms(cls_dets, cfg.TEST.NMS)

            """
            (Pdb) cls_dets.shape
            torch.Size([17, 5])
            """
            cls_dets = cls_dets[keep.view(-1).long()]
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array
      
      # Limit to max_per_image detections *over all classes*
      # phase 3
      curr_boxes = []
      curr_scores = []
      if max_per_image > 0:
          # flatten scores for all boxes of this image
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              # threshold to obtain max_per_image
              image_thresh = np.sort(image_scores)[-max_per_image]

              # for each class, extract boxes > threshold 
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      image_summary.pred.cls_prob = image_scores

      # Done nms on bboxes
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if(i % 100 == 0):
        tqdm.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))

      image_summary.pred.bbox_nms = [all_boxes[j][i] for j in range(imdb.num_classes)] # bboxes after nms
      image_summary.pred.scores_nms = vgg_extractor._detach2numpy(cls_scores) #### for all boxes? inspect

      feature_file = feature_folder + str(image_summary.info.image_idx) + ".pkl"
      with open(feature_file, 'wb') as f:
          pickle.dump(image_summary, f, pickle.HIGHEST_PROTOCOL)



  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
