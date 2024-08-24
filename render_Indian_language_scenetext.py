import os,sys,glob
from numpy.random import choice, random
import PIL
from PIL import Image
from tqdm import tqdm

with open(sys.argv[2]) as f:
	fontsList = f.read().splitlines()
print('number of unique fonts being considered= ', len(fontsList))
natural_bg_list = glob.glob("./dataset utils/background_images/natural_scenes/*")
plain_bg_list =glob.glob("./dataset utils/background_images/plain_colors/*")
background_type = ('natural', 'plain')
background_prob = [0.35, 0.65]

i_t_font = 'Noto Sans'

writeDirParent=sys.argv[3]+'/'
with open(sys.argv[1], 'r', encoding='utf8') as fp:
    words = [item.strip().split() for item in fp.readlines()]
distortArcOptions={'40','60','70','80','40','40','30'}
distorArcBooleanOptions={0,0,0,0,0,0,0,0,1}
densityOptions={'100','150','150','150','200','200','200','250','250','300','300','200','150','200','300','300','300','250','250'}
boldBooleanOptions={0,0,1}
italicBooleanOptions={0,0,0,0,0,0,0,0,0,0,0,0,1}
fontSizeOptions={'12','14','16','18','20','22','24','26','28','32','34','36'}
fontSizeOptions={'44','46','56','58','66','72'}
trimOptions={0,1}
fontStretchOptions={ 'semicondensed', 'normal', 'semiexpanded',  'normal', 'normal', 'normal','normal','normal','normal', 'normal','normal','normal','semicondensed','semicondensed','semicondensed','semicondensed','semicondensed', 'semiexpanded', 'semiexpanded', 'semiexpanded', 'semiexpanded','normal', 'normal', 'normal','normal','normal','normal', 'normal','normal','normal'}
shadowBooleanOptions={0,0,0,0,0,0,0,0,1}
perspectiveBooleanOptions={0,0,0,1,1,1,1}

shadowWidthOptions={'0','0','0','2','3','4'}
shadowSigmaOptions={'1','3'}
shadowOpacityOptions={'100','100','100','100','90','80','70'}
shadowWidthSignOptions={'+','-'}

numWords=len(words)
print('number of words in the vocab= ', numWords)

i_t_text_image_size = '256x100'
image_size = '256x128'
_img_size = image_size.split('x')
_img_size = [int(_img_size[0]), int(_img_size[1])]

num_samples = int(sys.argv[4])
for i in tqdm(range(0, num_samples), desc='Completed generations'):
    try:
        writeDir = os.path.join(writeDirParent, str(i))
        if not os.path.exists(writeDir):
            os.makedirs(writeDir)

        textImageName=str(i)+'_text.png'
        input_style_name = f'i_s_{i}.png'
        input_text_name = f'i_t_{i}.png'
        text_skeleton_name = f't_sk_1_{i}.png'
        fg_name = f't_t_{i}.png'
        bg_name = f't_b_{i}.png'
        final_style_name = f't_f_{i}.png'
        _tgt_text_fg_name = f'temp_tgt_text_fg_{i}.png'
        _src_text_fg_name = f'temp_src_text_fg_{i}.png'
        _input_text_fg_name = f'temp_input_text_fg_{i}.png'
        _skeleton_text_fg_name = f'temp_skeleton_text_fg_{i}.png'
        _black_bg_name = f'temp_black_bg_{i}.png'
        _gray_bg_name = f'temp_gray_bg_{i}.png'

        fg=random.sample(range(0, 255), 3)

        bg=random.sample(range(0, 255), 3)
        bg[0]=abs(fg[0]+100-255)
        bg[1]=abs(fg[0]+100-255)
        bg[2]=abs(fg[2]+125-255)
        sd=random.sample(range(0, 255), 3)

        fg_hex='#%02x%02x%02x' % (fg[0], fg[1], fg[2])
        bg_hex= '#%02x%02x%02x' % (bg[0], bg[1], bg[2])
        sd_hex= '#%02x%02x%02x' % (sd[0], sd[1], sd[2])

        if bool(random.getrandbits(1)):
            tmp=fg_hex
            fg_hex=bg_hex
            bg_hex=tmp

        density=random.sample(densityOptions,1)[0]
        distortArcBoolean=random.sample(distorArcBooleanOptions,1)[0]
        boldBoolean=random.sample(boldBooleanOptions,1)[0]
        italicBoolean=random.sample(italicBooleanOptions,1)[0]
        fontSize=random.sample(fontSizeOptions,1)[0]
        fontName=random.sample(fontsList,1)[0]
        fontStretch=random.sample(fontStretchOptions,1)[0]

        shadowOpacity=random.sample(shadowOpacityOptions,1)[0]
        shadowSigma=random.sample(shadowSigmaOptions,1)[0]
        ShadowWidth=random.sample(shadowWidthOptions,1)[0]
        ShadowWidthSign=random.sample(shadowWidthSignOptions,1)[0]

        perspectiveBoolean=random.sample(perspectiveBooleanOptions,1)[0]
        if perspectiveBoolean==1:
            sx=random.uniform(0.7, 1.3)
            ry=random.uniform(-0.8, 0.8)
            rx=random.uniform(-0.15, 0.15)
            sy=random.uniform(0.7, 1.3)
            px=random.uniform(0.0001, 0.001)
            py=random.uniform(0.0001, 0.001)

        chosen_background_type = choice(background_type, 1, p=background_prob).item()
        if chosen_background_type == 'natural':
            naturalImageName=random.sample(natural_bg_list,1)[0]
        elif chosen_background_type == 'plain':
            naturalImageName=random.sample(plain_bg_list,1)[0]

        grayImageName = './dataset utils/plain_colors/1280x800-roman-silver-solid-color-background.jpg'
        blackImageName = './dataset utils/plain_colors/1280x800-black-solid-color-background.jpg'

        if distortArcBoolean==1:
            distortArc=random.sample(distortArcOptions,1)[0]
        trimBoolean=random.sample(trimOptions,1)[0]
        word_idxs = random.sample(range(len(words)), 2)
        selected_eng_pair = [words[word_idxs[0]][0], words[word_idxs[1]][1]]
        
        labels = []
        
        for lang, textWord in zip(['eng', 'hin'], selected_eng_pair):
            command='convert -alpha set -background none'
            skewValue='0'
            arcValue='0'
            if distortArcBoolean==1:
                command+=' -distort Arc '+ distortArc
                arcValue=distortArc
            command+=' pango:\'   <span '
            command+='font_stretch='+'\"'+fontStretch+'\" '
            sk_command = command
            command+='foreground='+'\"'+fg_hex+'\" '
            sk_command+='foreground='+'\"'+'#ffffff'+'\" '
            labels.append(textWord)
            if lang == 'hin':
                with open(os.path.join(writeDir, 'labels.txt'), 'w') as fp:
                    fp.write(f'{labels[0]} {labels[1]}')
            
            origTextWord = textWord
            skTextWord = textWord
            if italicBoolean==1:
                textWord='<i>'+textWord+'</i>'
                skTextWord='<i>'+textWord+'</i>'
            if boldBoolean==1:
                textWord='<b>'+textWord+'</b>'
            fontString='font='+'\"'+fontName+' '+fontSize+' \">  '
            fontString+=' '+ textWord + '</span> \''
            command+=fontString
            sk_command+='weight=\"ultralight\" font='+'\"'+fontName+' '+fontSize+' \">  ' + ' '+ skTextWord + '</span> \''
            command+=' png:-|'
            sk_command+=' png:-|'

            command+='convert - ' + ' \\( +clone -background ' + '\''+str(sd_hex)+'\' -shadow '
            command+= shadowOpacity+'x'+shadowSigma+ShadowWidthSign+ShadowWidth+ShadowWidthSign+ShadowWidth + ' \\) +swap  -background none   -layers merge  +repage '+ 'png:-| '
            sk_command+='convert - +swap  -background none   -layers merge  +repage '+ 'png:-| '

            if perspectiveBoolean==1:
                command+='convert - ' + ' -alpha set -virtual-pixel transparent +distort Perspective-Projection '
                command+= '\''+str(sx)+ ', ' + str(ry) + ', 1.0\t' + str(rx) + ', ' + str(sy) + ', 1.0\t' + str(px) + ', ' + str(py) + '\'  png:-| '
                sk_command+='convert - ' + ' -alpha set -virtual-pixel transparent +distort Perspective-Projection '
                sk_command+= '\''+str(sx)+ ', ' + str(ry) + ', 1.0\t' + str(rx) + ', ' + str(sy) + ', 1.0\t' + str(px) + ', ' + str(py) + '\'  png:-| '
            command+= ' convert - '
            sk_command+= ' convert - '
            if trimBoolean==1:
                command+='  -trim '
                sk_command+='  -trim '
            command+=f' -resize {image_size} '
            if lang == 'eng':
                command+= os.path.join(writeDir, _src_text_fg_name)
            else:
                command+= os.path.join(writeDir, _tgt_text_fg_name)
            sk_command+=f' -resize {image_size} '
            sk_command+=os.path.join(writeDir, _skeleton_text_fg_name)

            os.system(command.encode('utf-8'))
            if lang == 'hin':
                os.system(sk_command.encode('utf-8'))

            if lang == 'hin':
                _input_text_fg_name = os.path.join(writeDir, _input_text_fg_name)
                input_text_command = f'convert -alpha set -background "rgb(121,127,141)" pango:\'  \
                    <span font_stretch="semicondensed" foreground="#000000" font=" {i_t_font} 30 ">{origTextWord}</span> \
                    \' \png:-|convert -  \\( +clone \\) +swap  -background "rgb(121,127,141)" -layers merge  +repage png:-|\
                    convert -   -trim  -resize {i_t_text_image_size} {_input_text_fg_name}'
                os.system(input_text_command.encode('utf-8'))

            if lang == 'eng':
                finalFgLayerName = os.path.join(writeDir, _src_text_fg_name)
            else:
                finalFgLayerName = os.path.join(writeDir, _tgt_text_fg_name)
            finakSkeleLayerName = os.path.join(writeDir, _skeleton_text_fg_name)
            im=Image.open(finalFgLayerName)
            imWidth, imHeight = _img_size

            if lang == 'eng':
                fgorBgBooleanOptions={0,1}
                fgOrBgBoolean=1
                fgImage=Image.open(naturalImageName)
                grayImage=Image.open(grayImageName)
                blackImage=Image.open(blackImageName)
                fgWidth, fgHeight = fgImage.size
                if fgWidth < imWidth+5 or  fgHeight < imHeight+5:
                    fgImage=fgImage.resize((imWidth+10,imHeight+10),PIL.Image.LANCZOS)
                    fgWidth, fgHeight = fgImage.size
                x=random.sample(range(0,fgWidth-imWidth+2 ),1)[0]
                y=random.sample(range(0, fgHeight-imHeight+2),1)[0]
                w=imWidth
                h=imHeight
                fgImageCrop=fgImage.crop((x ,y ,x+w, y+h))
                fgImageCropName=os.path.join(writeDir, bg_name)
                fgImageCrop.save(fgImageCropName)
            if lang == 'hin':
                grayImageCrop=grayImage.crop((x ,y ,x+w, y+h))
                grayImageCropName = os.path.join(writeDir, _gray_bg_name)
                os.system(f'convert -size {image_size} xc:"rgb(121, 127, 141)" {grayImageCropName}')
                
                blackImageCrop=blackImage.crop((x ,y ,x+w, y+h))
                blackImageCropName = os.path.join(writeDir, _black_bg_name)
                blackImageCrop.save(blackImageCropName)
            fgBlendBoolean=0
            bgNaturalImage=0	
            if fgOrBgBoolean==0:
                pass
            else:
                bgLayerName=fgImageCropName
                bgNaturalImage=1
            finalBgLayerName=bgLayerName
            if lang == 'eng':
                finalBlendImageName= os.path.join(writeDir, input_style_name)
            else:
                finalBlendImageName= os.path.join(writeDir, final_style_name)
            finalBlendCommand='composite -gravity center ' + finalFgLayerName + ' ' +  finalBgLayerName
            finalBlendCommand+=' png:-|'
            finalBlendCommand+='convert -  ' + finalBlendImageName
            os.system(finalBlendCommand.encode('utf-8'))

            if lang == 'hin':
                finalForegroundImageName = os.path.join(writeDir, fg_name)
                finalForegroundCommand='composite -gravity center ' + finalFgLayerName + ' ' +  grayImageCropName
                finalForegroundCommand+=' png:-|'
                finalForegroundCommand+='convert -  ' + finalForegroundImageName
                os.system(finalForegroundCommand.encode('utf-8'))

                finalInputTextImageName = os.path.join(writeDir, input_text_name)
                finalInputTextCommand='composite -gravity center ' + _input_text_fg_name + ' ' +  grayImageCropName
                finalInputTextCommand+=' png:-|'
                finalInputTextCommand+='convert -  ' + finalInputTextImageName
                os.system(finalInputTextCommand.encode('utf-8'))

                finalSkeletonImageName = os.path.join(writeDir, text_skeleton_name)
                finalSkeletonCommand='composite -gravity center ' + finakSkeleLayerName + ' ' +  blackImageCropName
                finalSkeletonCommand+=' png:-|'
                finalSkeletonCommand+='convert -  ' + finalSkeletonImageName
                os.system(finalSkeletonCommand.encode('utf-8'))
    except Exception as e:
        print(f'* * * * * Failed: {i} * * * * *')
        print(e)
        continue
