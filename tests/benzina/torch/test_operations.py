import numpy as np

import benzina.torch.operations as ops


def test_random_resized_crop():
    random_resized_crop = ops.RandomResizedCrop()
    similarity = ops.SimilarityTransform(scale=(0.08, 1.0),
                                         ratio=(3./4., 4./3.),
                                         random_crop=True)

    assert random_resized_crop.s != None
    assert random_resized_crop.s != (1.0, 1.0)
    assert random_resized_crop.ar != None
    assert random_resized_crop.resize
    assert random_resized_crop.random_crop

    assert random_resized_crop.s == similarity.s
    assert random_resized_crop.ar == similarity.ar
    assert random_resized_crop.r == similarity.r
    assert random_resized_crop.t == similarity.t
    assert random_resized_crop.fh == similarity.fh
    assert random_resized_crop.fv == similarity.fv
    assert random_resized_crop.resize == similarity.resize
    assert random_resized_crop.keep_ratio == similarity.keep_ratio
    assert random_resized_crop.random_crop == similarity.random_crop


def test_center_resized_crop():
    rng = np.random.RandomState(0)

    center_resized_crop = ops.CenterResizedCrop(0.5)
    similarity = ops.SimilarityTransform(scale=(0.25, 0.25), resize=True,
                                         keep_ratio=True)

    assert center_resized_crop.s != None
    assert center_resized_crop.s != (1.0, 1.0)
    assert center_resized_crop.keep_ratio

    assert center_resized_crop.s == similarity.s
    assert center_resized_crop.ar == similarity.ar
    assert center_resized_crop.r == similarity.r
    assert center_resized_crop.t == similarity.t
    assert center_resized_crop.fh == similarity.fh
    assert center_resized_crop.fv == similarity.fv
    assert center_resized_crop.resize == similarity.resize
    assert center_resized_crop.keep_ratio == similarity.keep_ratio
    assert center_resized_crop.random_crop == similarity.random_crop


def test_similarity_transform_vanilla():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform()

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert sim_t == (1, 0, 0,
                     0, 1, 0,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (0, 0, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) ==
            (255, 143, 1)).all()

    sim_t = similarity(0, (256, 144), (224, 126), rng)
    assert sim_t == (1, 0, 16,
                     0, 1, 9,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (16, 9, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((223, 125, 1)) ==
            (239, 134, 1)).all()

    sim_t = similarity(0, (256, 144), (144, 256), rng)
    assert sim_t == (1, 0, 56,
                     0, 1, -56,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (56, -56, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((143, 255, 1)) ==
            (199, 199, 1)).all()

    sim_t = similarity(0, (256, 144), (126, 224), rng)
    assert sim_t == (1, 0, 65,
                     0, 1, -40,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (65, -40, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((125, 223, 1)) ==
            (190, 183, 1)).all()


def test_similarity_transform_scale():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(scale=(0.25, 0.25))

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (63.75, 35.75, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) -
                        (191.25, 107.25, 1)) < 0.0001).all()

    sim_t = similarity(0, (256, 144), (224, 126), rng)
    o_top_left = np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1))
    o_bot_right = np.asarray(sim_t).reshape((3, 3)).dot((223, 125, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Crop scale is 0.5 and resize factors are 256/224 and 144/126 respectively
    assert abs(o_width * 224/256) * 2 + 1 - 224 < 0.0001
    assert abs(o_height * 126/144) * 2 + 1 - 126 < 0.0001


def test_similarity_transform_ratio():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(ratio=(16 / 9, 16 / 9))

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert sim_t == (1, 0, 0,
                     0, 1, 0,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (0, 0, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) ==
            (255, 143, 1)).all()

    similarity = ops.SimilarityTransform(ratio=(2, 2))

    sim_t = similarity(0, (512, 144), (288, 144), rng)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (112, 0, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((287, 143, 1)) ==
            (399, 143, 1)).all()

    sim_t = similarity(0, (512, 144), (144, 288), rng)
    # x: (288/144) * 0.5 - 0.5 = 0.5
    # y: (144/288) * 0.5 - 0.5 = -0.25
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (112.5, -0.25, 1)) < 0.0001).all()
    # x: -1 * (288/144) + 0.5 = -1.5
    # y: -1 * (144/288) + -0.25 = -0.75
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((143, 287, 1)) -
                        (398.5, 143.25, 1)) < 0.0001).all()

    similarity = ops.SimilarityTransform(ratio=(0.5, 0.5))

    sim_t = similarity(0, (144, 512), (288, 144), rng)
    # x: (144/288) * 0.5 - 0.5 = -0.25
    # y: (288/144) * 0.5 - 0.5 = 0.5
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (-0.25, 112.5, 1)).all()
    # x: -1 * (144/288) + -0.25 = -0.75
    # y: -1 * (288/144) + 0.5 = -1.5
    assert (np.asarray(sim_t).reshape((3, 3)).dot((287, 143, 1)) ==
            (143.25, 398.5, 1)).all()


def test_similarity_transform_degrees():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(degrees=(90, 90))

    sim_t = similarity(0, (256, 144), (144, 256), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (255, 0, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((143, 255, 1)) -
                        (0, 143, 1)) < 0.0001).all()

    sim_t = similarity(0, (256, 144), (72, 128), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (191, 36, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((71, 127, 1)) -
                        (64, 107, 1)) < 0.0001).all()

    similarity = ops.SimilarityTransform(degrees=(-90, -90))

    sim_t = similarity(0, (256, 144), (144, 256), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (0, 143, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((143, 255, 1)) -
                        (255, 0, 1)) < 0.0001).all()

    sim_t = similarity(0, (256, 144), (72, 128), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (64, 107, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((71, 127, 1)) -
                        (191, 36, 1)) < 0.0001).all()

    similarity = ops.SimilarityTransform(degrees=(180, 180))

    sim_t = similarity(0, (256, 144), (144, 256), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (199, 199, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((143, 255, 1)) -
                        (56, -56, 1)) < 0.0001).all()

    similarity = ops.SimilarityTransform(degrees=(180, 180))

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        (255, 143, 1)) < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) -
                        (0, 0, 1)) < 0.0001).all()


def test_similarity_transform_translate():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(translate=(0.1, 0.1))

    sim_t_samples = np.asarray([similarity(0, (256, 144), (256, 144), rng)
                                for _ in range(10000)])
    samples_mean = sim_t_samples.mean(axis=0)
    assert ((samples_mean == (1, 0, None,
                              0, 1, None,
                              0, 0, 1)) ==
            (True, True, False,
             True, True, False,
             True, True, True)).all()
    assert abs(samples_mean[2]) < 0.01 * 2 * 256 * similarity.t[0]
    assert abs(samples_mean[5]) < 0.01 * 2 * 144 * similarity.t[0]
    assert sim_t_samples[:, 2].max() < 256 * similarity.t[0]
    assert sim_t_samples[:, 5].max() < 144 * similarity.t[0]
    assert sim_t_samples[:, 2].min() > -256 * similarity.t[0]
    assert sim_t_samples[:, 5].min() > -144 * similarity.t[0]

    sim_t_samples = np.asarray([similarity(0, (256, 144), (128, 72), rng)
                                for _ in range(10000)])
    samples_mean = sim_t_samples.mean(axis=0)
    assert ((samples_mean == (1, 0, None,
                              0, 1, None,
                              0, 0, 1)) ==
            (True, True, False,
             True, True, False,
             True, True, True)).all()
    assert abs(samples_mean[2] - 64) < 0.01 * 2 * 128 * similarity.t[0]
    assert abs(samples_mean[5] - 36) < 0.01 * 2 * 72 * similarity.t[0]
    assert sim_t_samples[:, 2].max() - 64 < 128 * similarity.t[0]
    assert sim_t_samples[:, 5].max() - 36 < 72 * similarity.t[0]
    assert sim_t_samples[:, 2].min() + 64 > -128 * similarity.t[0]
    assert sim_t_samples[:, 5].min() + 36 > -72 * similarity.t[0]

    similarity = ops.SimilarityTransform(translate=(0.2, 0.2))

    sim_t_samples = np.asarray([similarity(0, (256, 144), (256, 144), rng)
                                for _ in range(10000)])
    samples_mean = sim_t_samples.mean(axis=0)
    assert ((samples_mean == (1, 0, None,
                              0, 1, None,
                              0, 0, 1)) ==
            (True, True, False,
             True, True, False,
             True, True, True)).all()
    assert abs(samples_mean[2]) < 0.01 * 2 * 256 * similarity.t[0]
    assert abs(samples_mean[5]) < 0.01 * 2 * 144 * similarity.t[0]
    assert sim_t_samples[:, 2].max() < 256 * similarity.t[0]
    assert sim_t_samples[:, 5].max() < 144 * similarity.t[0]
    assert sim_t_samples[:, 2].min() > -256 * similarity.t[0]
    assert sim_t_samples[:, 5].min() > -144 * similarity.t[0]

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (256, 144),
                                      translate=(10, -10))
    assert (sim_t == ((1, 0, 10),
                      (0, 1, -10),
                      (0, 0, 1))).all()
    assert (sim_t.dot((0, 0, 1)) == (10, -10, 1)).all()
    assert (sim_t.dot((255, 143, 1)) == (265, 133, 1)).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (128, 72),
                                      translate=(10, -10))
    assert (sim_t == ((1, 0, 74),
                      (0, 1, 26),
                      (0, 0, 1))).all()
    assert (sim_t.dot((0, 0, 1)) == (74, 26, 1)).all()
    assert (sim_t.dot((127, 71, 1)) == (201, 97, 1)).all()


def test_similarity_transform_flip():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(flip_h=1.0)

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert sim_t == (-1, 0, 255,
                     0, 1, 0,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (255, 0, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) ==
            (0, 143, 1)).all()

    similarity = ops.SimilarityTransform(flip_h=1.0)

    sim_t = similarity(0, (256, 144), (128, 72), rng)
    assert sim_t == (-1, 0, 191,
                     0, 1, 36,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (191, 36, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((127, 71, 1)) ==
            (64, 107, 1)).all()

    similarity = ops.SimilarityTransform(flip_v=1.0)

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert sim_t == (1, 0, 0,
                     0, -1, 143,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (0, 143, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) ==
            (255, 0, 1)).all()

    similarity = ops.SimilarityTransform(flip_v=1.0)

    sim_t = similarity(0, (256, 144), (128, 72), rng)
    assert sim_t == (1, 0, 64,
                     0, -1, 107,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (64, 107, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((127, 71, 1)) ==
            (191, 36, 1)).all()

    similarity = ops.SimilarityTransform(flip_h=1.0, flip_v=1.0)

    sim_t = similarity(0, (256, 144), (128, 72), rng)
    assert sim_t == (-1, 0, 191,
                     0, -1, 107,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (191, 107, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((127, 71, 1)) ==
            (64, 36, 1)).all()


def test_similarity_transform_resize():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(resize=True)

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert sim_t == (1, 0, 0,
                     0, 1, 0,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (0, 0, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) ==
            (255, 143, 1)).all()

    sim_t = similarity(0, (256, 144), (144, 256), rng)
    o_top_left = np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1))
    o_bot_right = np.asarray(sim_t).reshape((3, 3)).dot((143, 255, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Resize factors are 256/144 and 144/256 respectively
    assert abs(o_width * 144/256) + 1 - 144 < 0.0001
    assert abs(o_height * 256/144) + 1 - 256 < 0.0001


def test_similarity_transform_keep_ratio():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(keep_ratio=True)
    similarity_other = ops.SimilarityTransform()

    sim_t = similarity(0, (512, 288), (256, 256), rng)
    sim_t_other = similarity_other(0, (512, 288), (256, 256), rng)

    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) -
                        np.asarray(sim_t_other).reshape((3, 3)).dot((0, 0, 1)))
            < 0.0001).all()
    assert (np.absolute(np.asarray(sim_t).reshape((3, 3)).dot((255, 255, 1)) -
                        np.asarray(sim_t_other).reshape((3, 3))
                        .dot((255, 255, 1)))
            < 0.0001).all()

    similarity = ops.SimilarityTransform(keep_ratio=True,
                                         resize=True)

    sim_t = similarity(0, (512, 288), (256, 256), rng)
    o_top_left = np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1))
    o_bot_right = np.asarray(sim_t).reshape((3, 3)).dot((255, 255, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Resize factors are 288/256
    assert (o_width * 256/288) + 1 - 256 < 0.0001
    assert (o_height * 256/288) + 1 - 256 < 0.0001


def test_similarity_transform_crop():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(random_crop=True)

    sim_t = similarity(0, (256, 144), (256, 144), rng)
    assert sim_t == (1, 0, 0,
                     0, 1, 0,
                     0, 0, 1)
    assert (np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1)) ==
            (0, 0, 1)).all()
    assert (np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1)) ==
            (255, 143, 1)).all()

    sim_t_samples = np.asarray([similarity(0, (256, 144), (128, 72), rng)
                                for _ in range(10000)])
    samples_mean = sim_t_samples.mean(axis=0)
    assert ((samples_mean == (1, 0, None,
                              0, 1, None,
                              0, 0, 1)) ==
            (True, True, False,
             True, True, False,
             True, True, True)).all()
    assert abs(samples_mean[2]) - 64 < 0.01 * 128
    assert abs(samples_mean[5]) - 36 < 0.01 * 72
    assert sim_t_samples[:, 2].max() < 128
    assert sim_t_samples[:, 5].max() < 72
    assert sim_t_samples[:, 2].min() >= 0
    assert sim_t_samples[:, 5].min() >= 0

    sim_t_samples = np.asarray([similarity(0, (256, 144), (144, 256), rng)
                                for _ in range(10000)])
    samples_mean = sim_t_samples.mean(axis=0)
    assert ((samples_mean == (1, 0, None,
                              0, 1, -56,
                              0, 0, 1)) ==
            (True, True, False,
             True, True, True,
             True, True, True)).all()
    assert abs(samples_mean[2]) - 56 < 0.01 * 112
    assert sim_t_samples[:, 2].max() <= 112
    assert sim_t_samples[:, 2].min() >= 0

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (256, 144),
                                      crop=(10, -10, 40, 20))
    assert (sim_t == ((1, 0, 10),
                      (0, 1, -10),
                      (0, 0, 1))).all()
    assert (sim_t.dot((0, 0, 1)) == (10, -10, 1)).all()
    assert (sim_t.dot((255, 143, 1)) == (265, 133, 1)).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (128, 72),
                                      crop=(10, -10, 40, 20))
    assert (sim_t == ((1, 0, 74),
                      (0, 1, 26),
                      (0, 0, 1))).all()
    assert (sim_t.dot((0, 0, 1)) == (74, 26, 1)).all()
    assert (sim_t.dot((127, 71, 1)) == (201, 97, 1)).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (256, 144),
                                      crop=(10, -10, 40, 20),
                                      resize=True)
    o_top_left = sim_t.dot((0, 0, 1))
    o_bot_right = sim_t.dot((255, 143, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Resize factors are 40/256 and 20/144 respectively
    assert abs(o_width * 256/40) + 1 - 256 < 0.0001
    assert abs(o_height * 144/20) + 1 - 144 < 0.0001

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (128, 72),
                                      crop=(10, -10, 40, 20),
                                      resize=True)
    o_top_left = sim_t.dot((0, 0, 1))
    o_bot_right = sim_t.dot((255, 143, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Resize factors are 40/128 and 20/72 respectively
    assert abs(o_width * 128/40) + 1 - 256 < 0.0001
    assert abs(o_height * 72/20) + 1 - 144 < 0.0001


def test_similarity_transform_mix():
    rng = np.random.RandomState(0)

    similarity = ops.SimilarityTransform(scale=(0.25, 0.25),
                                         ratio=(2, 2),
                                         degrees=(90, 90),
                                         flip_h=1.0,
                                         flip_v=1.0)

    sim_t = similarity(0, (512, 144), (192, 384), rng)
    o_top_left = np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1))
    o_bot_right = np.asarray(sim_t).reshape((3, 3)).dot((191, 383, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Crop scale is 0.5 and resize factors are 384/384 and 192/192
    assert abs(o_width) * 2 + 1 - 384 < 0.0001
    assert abs(o_height) * 2 + 1 - 192 < 0.0001

    similarity = ops.SimilarityTransform(scale=(0.0625, 0.0625),
                                         keep_ratio=True)

    sim_t = similarity(0, (768, 288), (256, 144), rng)
    o_top_left = np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1))
    o_bot_right = np.asarray(sim_t).reshape((3, 3)).dot((255, 143, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Crop scale is 0.25 and resize factors are 72/144
    assert abs(o_width * 144/72) + 1 - 256 < 0.0001
    assert abs(o_height * 144/72) + 1 - 144 < 0.0001

    similarity = ops.SimilarityTransform(scale=(0.0625, 0.0625),
                                         keep_ratio=True)

    sim_t = similarity(0, (768, 288), (144, 256), rng)
    o_top_left = np.asarray(sim_t).reshape((3, 3)).dot((0, 0, 1))
    o_bot_right = np.asarray(sim_t).reshape((3, 3)).dot((143, 255, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Crop scale is 0.25 and resize factors are 72/256
    assert abs(o_width * 256/72) + 1 - 144 < 0.0001
    assert abs(o_height * 256/72) + 1 - 256 < 0.0001

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (144, 256),
                                      crop=(10, -10, 40, 20),
                                      degrees=90)
    assert (np.absolute(sim_t.dot((0, 0, 1)) - (265, -10, 1)) < 0.0001).all()
    assert (np.absolute(sim_t.dot((143, 255, 1)) - (10, 133, 1))
            < 0.0001).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (144, 256),
                                      crop=(10, -10, 40, 20),
                                      degrees=90,
                                      resize=True)
    o_top_left = sim_t.dot((0, 0, 1))
    o_bot_right = sim_t.dot((255, 143, 1))
    o_width, o_height = (o_bot_right - o_top_left)[0:2]
    # Resize factors are 40/256 and 20/144 respectively
    assert abs(o_width * 256/40) + 1 - 144 < 0.0001
    assert abs(o_height * 144/20) + 1 - 256 < 0.0001

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (256, 144),
                                      crop=(10, -10, 40, 20),
                                      translate=(-15, 15))
    assert (sim_t == ((1, 0, -5),
                      (0, 1, 5),
                      (0, 0, 1))).all()
    assert (sim_t.dot((0, 0, 1)) == (-5, 5, 1)).all()
    assert (sim_t.dot((255, 143, 1)) == (250, 148, 1)).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (128, 72),
                                      crop=(10, -10, 40, 20),
                                      translate=(-15, 15))
    assert (sim_t == ((1, 0, 59),
                      (0, 1, 41),
                      (0, 0, 1))).all()
    assert (sim_t.dot((0, 0, 1)) == (59, 41, 1)).all()
    assert (sim_t.dot((127, 71, 1)) == (186, 112, 1)).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (256, 144),
                                      crop=(10, -10, 40, 20),
                                      translate=(-15, 15),
                                      resize=True)
    sim_t_other = ops.compute_affine_matrix((256, 144),
                                            (256, 144),
                                            crop=(10, -10, 40, 20),
                                            resize=True)
    assert (np.absolute(sim_t.dot((15, -15, 1)) - sim_t_other.dot((0, 0, 1)))
            < 0.0001).all()
    assert (np.absolute(sim_t.dot((270, 128, 1)) -
                        sim_t_other.dot((255, 143, 1)))
            < 0.0001).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (128, 72),
                                      crop=(10, -10, 40, 20),
                                      translate=(-15, 15),
                                      resize=True)
    sim_t_other = ops.compute_affine_matrix((256, 144),
                                            (128, 72),
                                            crop=(10, -10, 40, 20),
                                            resize=True)
    assert (np.absolute(sim_t.dot((15, -15, 1)) - sim_t_other.dot((0, 0, 1)))
            < 0.0001).all()
    assert (np.absolute(sim_t.dot((142, 56, 1)) -
                        sim_t_other.dot((127, 71, 1)))
            < 0.0001).all()

    sim_t = ops.compute_affine_matrix((256, 144),
                                      (144, 256),
                                      crop=(10, -10, 40, 20),
                                      degrees=90,
                                      translate=(-15, 15),
                                      resize=True)
    sim_t_other = ops.compute_affine_matrix((256, 144),
                                            (144, 256),
                                            crop=(10, -10, 40, 20),
                                            degrees=90,
                                            resize=True)
    assert (np.absolute(sim_t.dot((15, -15, 1)) - sim_t_other.dot((0, 0, 1)))
            < 0.0001).all()
    assert (np.absolute(sim_t.dot((270, 128, 1)) -
                        sim_t_other.dot((255, 143, 1)))
            < 0.0001).all()
