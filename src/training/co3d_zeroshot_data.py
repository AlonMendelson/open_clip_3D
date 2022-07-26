

co3d_classnames = ["backpack","bicycle","book","car","chair","hairdryer","handbag","hydrant","keyboard","laptop","motorcycle","mouse","remote"
                   ,"teddybear", "toaster","toilet","toybus","toyplane","toytrain","toytruck"]





co3d_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of a {c} from the front.',
    lambda c: f'a photo of a {c} rotated by 18 degrees.',
    lambda c: f'a photo of a {c} rotated by 36 degrees.',
    lambda c: f'a photo of a {c} rotated by 54 degrees.',
    lambda c: f'a photo of a {c} rotated by 72 degrees.',
    lambda c: f'a photo of a {c} rotated by 90 degrees.',
    lambda c: f'a photo of a {c} rotated by 108 degrees.',
    lambda c: f'a photo of a {c} rotated by 126 degrees.',
    lambda c: f'a photo of a {c} rotated by 144 degrees.',
    lambda c: f'a photo of a {c} rotated by 162 degrees.'
]

co3d_loss_template = [lambda c: f'a photo of a {c}.']