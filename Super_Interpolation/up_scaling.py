from function import *

def up_scaling(test_img, lin_map):

    # initialize
    blur_img = cv2.blur(test_img, (3, 3))
    lr_img = cv2.resize(blur_img, (test_img.shape[1]//2, test_img.shape[0]//2), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)
    lr_img, cb, cr = cv2.split(lr_img)

    # padding
    pad_img = cv2.copyMakeBorder(lr_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # generate 3x3 LR patches
    lr_patches = get_patches(pad_img)

    # each LR input patch
    cls = []
    for lr_patch in lr_patches:
        # get sub patches
        sub_patches = get_sub_patches(lr_patch)

        # Each 2x2 LR sub-patches
        h_gradients = []
        v_gradients = []
        for sub_patch in sub_patches:
            h_gradient, v_gradient = calculate_gradient(sub_patch)
            h_gradients.append(h_gradient)
            v_gradients.append(v_gradient)

        # compute EO class index
        cls.append(compute_class(h_gradients, v_gradients))

    # look up corresponding linear mapping by its EO class index C
    hr_patches = []
    for lr_patch, cls_idx in zip(lr_patches, cls):
        m = lin_map[cls_idx] if len(lin_map[cls_idx]) > 0 else 0

        hr_patch = lr_patch @ m
        hr_patches.append(hr_patch)

    hr_patches = np.array(hr_patches, dtype=np.uint8)
    hr_patches = hr_patches.reshape((lr_img.shape[0]*2, lr_img.shape[1]*2))

    cb = cv2.resize(cb, (cb.shape[0]*2, cb.shape[1]*2), interpolation=cv2.INTER_CUBIC)
    cr = cv2.resize(cr, (cr.shape[0]*2, cr.shape[1]*2), interpolation=cv2.INTER_CUBIC)

    merged = cv2.merge([hr_patches, cb, cr])
    merged = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    # cv2.imshow('merged', merged)
    # cv2.imshow('blur_img', blur_img)
    # cv2.imshow('lr_img', lr_img)
    # cv2.imshow('cb', cb)
    # cv2.imshow('cr', cr)
    # cv2.imshow('test_img', test_img)
    # cv2.imshow('hr_patches', hr_patches)
    #
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return merged