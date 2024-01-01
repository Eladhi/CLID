from inference import *
from evaluate import *


def inference_wrapper(data_loader):
    save_dir = os.path.join(config.save_dir)
    mkdir(save_dir)
    logger = setup_logger("inference", save_dir, 0)
    logger.info("Running with config:\n{}".format(config))

    device = torch.device(config.device)
    num_types = len(config.boundaries) + 2

    generator = Generator(BertConfig(type_vocab_size=num_types))
    generator = generator.to(device)
    g_checkpointer = Checkpointer(model=generator, logger=logger)
    g_checkpointer.load(config.model_path, True)

    pred_dict = inference(generator, data_loader, device)
    logger.info(f"Saving results to {save_dir}/caption_results.json")
    with open(os.path.join(save_dir, 'caption_results.json'), 'w') as f:
        json.dump(pred_dict, f)


def eval(args):
    logger = setup_logger("evaluate", args.save_dir, 0)
    ptb_tokenizer = PTBTokenizer()

    scorers = [(Cider(), "C"), (Spice(), "S"),
               (Bleu(4), ["B1", "B2", "B3", "B4"]),
               (Meteor(), "M"), (Rouge(), "R")]

    logger.info(f"loading ground-truths from {args.gt_caption}")
    with open(args.gt_caption) as f:
        gt_captions = json.load(f)
    gt_captions = ptb_tokenizer.tokenize(gt_captions)

    logger.info(f"loading predictions from {args.pd_caption}")
    with open(args.pd_caption) as f:
        pred_dict = json.load(f)
    pd_captions = dict()
    for level, v in pred_dict.items():
        pd_captions[level] = ptb_tokenizer.tokenize(v)

    logger.info("Start evaluating")
    score_all_level = list()
    spice_all_level = list()
    b4_all_level = list()
    score_dict_all = {}
    for level, v in pd_captions.items():
        scores = {}
        for (scorer, method) in scorers:
            gt_captions_filtered = {i: gt_captions[i] for i in v.keys()}
            if len(v.keys()):
                score, score_list = scorer.compute_score(gt_captions_filtered, v)
            else:  # if there're no captions in this level
                if type(method) == list:  # for [B1,B2,B3,B4]
                    score, score_list = [0.0, 0.0, 0.0, 0.0], [[0.0], [0.0], [0.0], [0.0]]
                else:
                    score, score_list = 0.0, np.zeros(1)
            if type(score) == list:
                for m, s in zip(method, score):
                    scores[m] = s
                b4_all_level.append(np.asarray(score_list[3]))
            else:
                scores[method] = score
            if method == "C":
                score_all_level.append(np.asarray(score_list))
            elif method == "S":
                spice_all_level.append(np.asarray(score_list))

        logger.info(
            ' '.join([
                "C: {C:.4f}", "S: {S:.4f}",
                "M: {M:.4f}", "R: {R:.4f}",
                "B1: {B1:.4f}", "B2: {B2:.4f}",
                "B3: {B3:.4f}", "B4: {B4:.4f}"
            ]).format(
                C=scores['C'], S=scores['S'],
                M=scores['M'], R=scores['R'],
                B1=scores['B1'], B2=scores['B2'],
                B3=scores['B3'], B4=scores['B4']
            ))
        score_dict_all[level] = scores

    num_captions = [s.size for s in score_all_level]
    for i, s in enumerate(score_all_level):
        score_all_level[i] = np.pad(s, (0, 5000 - s.shape[0]), 'constant', constant_values=0.0)

    score_all_level = np.stack(score_all_level, axis=1)
    logger.info(
        '  '.join([
            "4 level ensemble CIDEr: {C4:.4f}",
            "3 level ensemble CIDEr: {C3:.4f}",
            "2 level ensemble CIDEr: {C2:.4f}",
        ]).format(
            C4=score_all_level.max(axis=1).mean(),
            C3=score_all_level[:, :3].max(axis=1).mean(),
            C2=score_all_level[:, :2].max(axis=1).mean(),
        ))
    return score_dict_all, num_captions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference_evaluate")
    parser.add_argument("--gt_caption", type=str)
    parser.add_argument("--pd_caption", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config.merge_from_list(args.opts)
    config.freeze()

    dataset = COCOCaptionDataset(
        root=config.data_dir,
        split='test',
        boundaries=config.boundaries,
        arguments=args
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=config.samples_per_gpu,
        num_workers=config.num_workers,
        split='test'
    )

    scores = []
    num_captions = []
    checkpoints = []
    filename = config['model_path']
    config['model_path'] = filename
    config.freeze()

    inference_wrapper(data_loader)  # performs the inference and saves captions in a json file
    sc, nc = eval(args)  # performs the evaluation, reading captions from the json file

    scores.append(sc)
    checkpoints.append(filename)
    num_captions.append(nc)

    with open(os.path.join(args.save_dir, 'score_comparisons.txt'), 'w') as f:
        for item, name, n_cap in zip(scores, checkpoints, num_captions):
            f.write('%s\n' % name)
            for level in item.keys():
                f.write('Level: ' + level + '\n')
                for elem in item[level]:
                    f.write('%s %.2f\n' % (elem, item[level][elem]*100))
            f.write('Captions per Level: ' + ' '.join([str(t) for t in n_cap]) + '\n')
            f.write('Precision per Level: ' + ' '.join([str(t/5000) for t in n_cap]) + '\n')
        f.write('\n')

    f.close()
