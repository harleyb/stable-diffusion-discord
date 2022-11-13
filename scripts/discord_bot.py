#!/usr/bin/env python3

import asyncio
import concurrent.futures
import functools
import argparse
import shlex
import os
import sys
import time
import copy
import traceback
import gc

import numpy as np
import random
import logging
import logging.handlers
import tempfile
import discord
from dotenv import load_dotenv
from PIL import Image
import torch

debugging = False

logger = logging.getLogger('SDDisc')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('discord_log.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

DEFAULT_ITERATIONS = 3
DISCORD_MESSAGE_LIMIT = 2000
HISTORY_COUNT = 10
NORM_DIM = 512
BIG_DIM = 768

SAMPLER_CHOICES = [
    'ddim',
    'k_euler_a',
    'k_dpm_2_a',
    'k_dpm_2',
    'k_euler',
    'k_heun',
    'k_lms',
    'plms',
]

SAMPLER_EMOJI = {
    'ddim': '1️⃣',
    'k_euler_a': '2️⃣',
    'k_dpm_2_a': '3️⃣',
    'k_dpm_2': '4️⃣',
    'k_euler': '5️⃣',
    'k_heun': '6️⃣',
    'k_lms': '7️⃣',
    'plms': '8️⃣',
}

PRECISION_CHOICES = [
    'auto',
    'float32',
    'autocast',
    'float16',
]


class ResourceError(Exception):
    pass


class DiscordBot(object):
    def __init__(self):
        self.argvopt = self.get_argv_parser().parse_args()
        self.gfpgan, self.codeformer, self.esrgan = None, None, None
        self.t2i = self.init_model()
        self.interrogator = self.init_interrogator()
        self.prompt_parser = self.get_prompt_parser()
        self.opt_history = list()
        self.threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        logging.basicConfig(
            handlers=[
                logging.FileHandler(os.path.join(self.argvopt.outdir, 'discord_log.txt')),
                logging.StreamHandler(sys.stdout)
            ],
            level=logging.DEBUG
        )

        load_dotenv()
        self.TOKEN = os.getenv('DISCORD_TOKEN')
        self.GUILD = int(os.getenv('DISCORD_GUILD'))
        self.intents = discord.Intents.default()
        self.intents.message_content = True

        self.client = discord.Client(intents=self.intents)

        @self.client.event
        async def on_ready():
            found_guild = False
            logger.info(f'{self.client.user} is connected to the following guild(s):\n')
            for guild in self.client.guilds:
                logger.info(f'{guild.name}(id: {guild.id})')
                if int(guild.id) == self.GUILD:
                    found_guild = True
            if not found_guild:
                logger.error(f'{self.client.user} is not connected to the specified guild')

        @self.client.event
        async def on_member_join(member):
            if not self.check_guild(member.guild):
                return
            await member.create_dm()
            await member.dm_channel.send("DM me to get help")

        @self.client.event
        async def on_reaction_add(reaction, user):
            if reaction.message.author != self.client.user:
                return

            inmoji = str(reaction.emoji)
            try:
                content = reaction.message.content.split(': ')[1].strip('`')
                opt = self.parse_command(content)
                prompt = opt.prompt
            except Exception as e:
                logger.error(f"couldn't evaluate {reaction.message.content}", exc_info=e)
                return

            if inmoji == '😍':  # make upscaled, up-stepped image
                if opt.sampler_name == 'ddim':
                    opt.sampler_name = 'plms'
                if opt.sampler_name in ['k_euler_a', 'k_dpm_2_a']:
                    pass
                else:
                    opt.steps = 64
                opt.gfpgan_strength = 0.8
                opt.upscale = [4, 0.7]
                cmd = [self.normalize_prompt(opt)]
                msg = 'generating (~1min) detailed'
            elif inmoji == '🤩':  # make make upscaled image with dpm_2_a sampler
                cmd = list()
                opt.sampler_name = 'k_dpm_2_a'
                opt.steps = 128
                opt.facetool_strength = 0.8
                opt.upscale = [4, 0.7]
                for y in [32, 48, 64]:
                    opt.steps = y
                    cmd.append(self.normalize_prompt(opt))
                msg = 'generating (~1min) a spread of Style 3️⃣ detailed'
            elif inmoji == '🤓' or inmoji == '😎':  # CFG spread
                cmd = list()
                for y in [2, 4.75, 12, 18]:
                    opt.cfg_scale = y
                    cmd.append(self.normalize_prompt(opt))
                msg = 'generating a strictness spread of'
            elif inmoji == '😨':  # perlin spread
                if opt.sampler_name in ['ddim', 'k_lms']:
                    opt.sampler_name = 'k_euler_a'
                cmd = list()
                for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    opt.perlin = y
                    cmd.append(self.normalize_prompt(opt))
                msg = 'generating a variation spread of'
            elif inmoji == '😱':  # get a k_euler_a spread
                cmd = list()
                opt.sampler_name = 'k_euler_a'
                for y in [12, 20, 28, 36, 48]:
                    opt.steps = y
                    cmd.append(self.normalize_prompt(opt))
                msg = 'generating a Style 2️⃣ step spread of'
            else:
                return

            try:
                opts = [self.parse_command(c) for c in cmd]
            except Exception as e:
                logger.error(f"couldn't evaluate {content}", exc_info=e)
                return
            await reaction.message.channel.send(f"{msg}: {prompt}...")
            # not appending these to the prompt history tracker
            # self.opt_history.append([copy.deepcopy(x) for x in opts])
            for opt in opts:
                await self.generator(opt, discord_channel=reaction.message.channel)

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return
            if not hasattr(message, "guild") or message.guild is None:
                # probably a DM; maybe a better way to check?
                await self.send_help_text(message.channel)
                return

            if len(message.content) > 0 and message.content[0] in ['!', '$', '%']:
                content = message.content[1:].lower().strip()
                do_history = content in ['history']
                do_help = content in ['help']
                do_more = False
                more_idx = None
                try:
                    split = [x for x in content.split(' ') if x]
                    if len(split) > 1:
                        more_idx = int(split[1])
                    elif len(split) > 2:
                        raise Exception
                    else:
                        more_idx = 1
                    do_more = split[0] in ['more']
                except Exception as e:
                    pass
                if do_help:
                    await self.send_help_text(message.channel, extended=False)
                    return
                elif do_history:
                    history_content = "Regenerate a previous prompt with `!more <x>` (or `$` or `%` for portrait or landscape) where `<x>` is the number:\n" + \
                                      "\n".join([f"\t**{idx}**: `{opt.prompt}`"
                                                 for idx, opt in enumerate(self.opt_history[-1:-HISTORY_COUNT:-1], 1)])
                    if len(history_content) > DISCORD_MESSAGE_LIMIT:
                        for line in history_content.split("\n"):
                            await message.channel.send(line)
                    else:
                        await message.channel.send(history_content)
                    return
                elif do_more:
                    if more_idx > len(self.opt_history) or more_idx < 1:
                        await message.channel.send(
                            f"couldn't generate, only have {len(self.opt_history)} history entries")
                        return
                    opt = copy.deepcopy(self.opt_history[-more_idx])
                    opt.seed = None
                else:
                    try:
                        opt = self.parse_command(message.content[1:])
                        self.opt_history.append(copy.deepcopy(opt))
                    except ResourceError as e:
                        await message.channel.send(f"couldn't generate, {e.args[0]}")
                        return
                    except Exception as e:
                        logger.error(f"couldn't evaluate {message}", exc_info=e)
                        await message.channel.send(f"couldn't evaluate {message.content[1:]}, DM me to get help")
                        return

                if message.content[0] == '$':
                    opt.height = BIG_DIM
                    opt.width = NORM_DIM
                elif message.content[0] == '%':
                    opt.height = NORM_DIM
                    opt.width = BIG_DIM
                await message.channel.send(f"generating {opt.prompt}...")
                if opt.sampler_name is None and opt.iterations is None and opt.seed is None:
                    opts = list()
                    for _ in range(DEFAULT_ITERATIONS):
                        seed = random.randrange(0, np.iinfo(np.uint32).max)
                        for y in ['ddim', 'k_euler_a']:
                            opt2 = copy.deepcopy(opt)
                            opt2.sampler_name = y
                            opt2.seed = seed
                            opts.append(opt2)
                    for opt in opts:
                        await self.generator(opt, discord_channel=message.channel)
                else:
                    if opt.seed is None:
                        opt.seed = random.randrange(0, np.iinfo(np.uint32).max)
                    await self.generator(opt, discord_channel=message.channel)
            elif len(message.attachments) > 0:
                # self.t2i.model = None
                # self.t2i.sampler = None
                # self.t2i.generators = {}
                try:
                    self.unload_everything()
                    for attachment in message.attachments:
                        await message.reply(f"Analyzing `{attachment.filename}`...")
                        result = await self.interrogate_attachment(attachment)
                        await message.reply(f"`{attachment.filename}`: \"{result}\"")
                finally:
                    self.reload_everything()

    def run(self):
        self.client.run(self.TOKEN)

    def check_guild(self, guild):
        return self.GUILD == int(guild.id)

    async def send_help_text(self, channel, extended=True):
        msg = f'''Send a message to any channel I'm in starting with an exclamation mark and I'll make images from the message body (the "prompt")! For example:
\t`!a spider wearing a hat` generates six images total: three different seeds, each generated with styles 1️⃣ and 2️⃣.
Use a `$` or a `%` instead of the `!` to generate portrait or landscape aspect ratios! (Note that seeds will generate completely differently in different ratios)
\t`$a beautiful portrait of princess peach`
\t`%a stunning panorama of the mushroom kingdom`
Use an emoji react to explore this prompt further! These will use the same seed value, which tends to keep the image composition similar.
\t😍: generate a more detailed version with the same seed, style and strictness (and steps if style 2️⃣) (~1min)
\t🤩: generate a spread of style 3️⃣ detailed images with the same seed and strictness (~1min)
\t🤓 or 😎: generate a spread using this seed and style, with the bot varying how closely it follows the prompt
\t😨: generate a spread using this seed with variations in initial noise during generation (style 1️⃣ will switch to 2️⃣) 
\t😱: generate a spread using this seed in style 2️⃣, with the bot varying the number of steps
\t⭐️: add to the #hall-of-fame channel
**Check the #general pinned messages for tips on crafting prompts!** This is the most important part!

Send `!history` to get the last {HISTORY_COUNT} prompts I've generated.

Send an image to a channel and I'll tell you what I think the prompt would be!

'''
        await channel.send(msg)

        if extended:
            msg = '''Using flags to customize generation:
`!a raccoon riding a bicycle -1 -U 2 0.75` enable upscaling at 0.75% smoothing strength
`!danny devito headshot -G0.5 -mddim` use the DDIM sampler with GFPGAN facial reconstruction at 50% smoothing strength
Flags available:
'''
            await channel.send(msg)
            msgs = str(self.prompt_parser.format_help().split('options:')[1]).split('\n')
            outmsg = ''
            for msg in msgs:
                if len(outmsg) + len(msg) > 2000:
                    await channel.send(outmsg)
                    outmsg = ''
                outmsg = outmsg + msg + '\n'
            await channel.send(outmsg)

    def get_argv_parser(self):
        parser = argparse.ArgumentParser(
            description="Parse script's command line args",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '-F', '--full_precision',
            dest='full_precision',
            action='store_true',
            help="use more memory-intensive full precision math for calculations")
        parser.add_argument(
            '-A',
            '-m',
            '--sampler',
            dest='sampler_name',
            choices=SAMPLER_CHOICES,
            metavar='SAMPLER_NAME',
            default='k_lms',
            help='Which sampler to use',
        )
        parser.add_argument(
            '--outdir',
            '-o',
            type=str,
            default="outputs/img-samples",
            help="directory in which to place generated images and a log of prompts and seeds")
        parser.add_argument(
            '--embedding_path',
            type=str,
            help="Path to a pre-trained embedding manager checkpoint - can only be set on command line")
        parser.add_argument(
            '--device',
            '-d',
            type=str,
            default="cuda",
            help="device to run stable diffusion on. defaults to cuda `torch.cuda.current_device()` if available")
        parser.add_argument(
            '--model',
            help='Indicates which diffusion model to load. (defaults to "default" stanza in configs/models.yaml)',
        )
        parser.add_argument(
            '--config',
            default='./configs/models.yaml',
            help='Path to configuration file for alternate models.',
        )
        parser.add_argument(
            '--precision',
            dest='precision',
            type=str,
            choices=PRECISION_CHOICES,
            metavar='PRECISION',
            help=f'Set model precision. Defaults to auto selected based on device.',
            default='auto',
        )
        parser.add_argument(
            '--max_loaded_models',
            dest='max_loaded_models',
            type=int,
            default=2,
            help='Maximum number of models to keep in memory for fast switching, including the one in GPU',
        )
        parser.add_argument(
            '--free_gpu_mem',
            dest='free_gpu_mem',
            action='store_true',
            help='Force free gpu memory before final decoding',
        )
        parser.add_argument(
            '--esrgan_bg_tile',
            type=int,
            default=400,
            help='Tile size for background sampler, 0 for no tile during testing. Default: 400.',
        )
        parser.add_argument(
            '--gfpgan_model_path',
            type=str,
            default='./models/gfpgan/GFPGANv1.4.pth',
            help='indicates the path to the GFPGAN model',
        )
        return parser

    def get_prompt_parser(self):
        parser = argparse.ArgumentParser(
            description='Example: !a fantastic alien landscape -s10 -n5',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('prompt')
        parser.add_argument(
            '-s',
            '--steps',
            type=int,
            help="number of steps")
        parser.add_argument(
            '-S',
            '--seed',
            type=int,
            help='image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc')
        parser.add_argument(
            '-n',
            '--iterations',
            type=int,
            default=None,
            help="number of samplings to perform (slower, but will provide seeds for individual images)")
        parser.add_argument(
            '-C', '--cfg_scale',
            default=7.5,
            type=float,
            help="Classifier free guidance (CFG) scale - higher numbers cause generator to 'try' harder.")
        # TODO: handle url for img2img with results = t2i.img2img(**vars(opt))
        parser.add_argument(
            '-G',
            '--facetool_strength',
            default=None,
            type=float,
            help='The strength at which to apply the GFPGAN model to the result. Set to <0 to disable',
        )
        parser.add_argument(
            '-U',
            '--upscale',
            nargs='+',
            default=None,
            type=float,
            help='Scale factor (2, 4) for upscaling followed by upscaling strength (0-1.0). If strength not specified, defaults to 0.75. set to <0 to disable'
        )
        parser.add_argument(
            '-A',
            '-m',
            '--sampler',
            dest='sampler_name',
            default=None,
            type=str,
            choices=SAMPLER_CHOICES,
            metavar='SAMPLER_NAME',
            help='Change to another supported sampler',
        )
        parser.add_argument(
            '-W', '--width', type=int, help='Image width, multiple of 64'
        )
        parser.add_argument(
            '-H', '--height', type=int, help='Image height, multiple of 64'
        )
        parser.add_argument(
            '-l'
            '--seamless',
            dest='seamless',
            action='store_true',
            help='Change the model to seamless tiling (circular) mode',
        )
        parser.add_argument(
            '--perlin',
            default=0.0,
            type=float,
            help='Perlin noise scale (0.0 - 1.0) - add perlin noise to the initialization instead of the usual gaussian noise.',
        )
        return parser

    async def interrogate_attachment(self, attachment):
        result = "error processing"
        fp = tempfile.NamedTemporaryFile(delete=False)
        fp.close()
        try:
            await attachment.save(fp.name, use_cached=True)
            fp.close()
            result = self.interrogator.interrogate(Image.open(fp.name))
        except:
            traceback.print_exc()
        finally:
            os.unlink(fp.name)
        return result

    def init_interrogator(self):
        from clip_interrogator import Interrogator, Config
        ci = Interrogator(Config(clip_model_name="ViT-B/16", device="cuda:0"))
        return ci

    def init_model(self):
        ''' Initialize command-line parsers and the diffusion model '''

        logger.info("* Initializing, be patient...")
        sys.path.append('.')
        from pytorch_lightning import logging as pytorch_logging
        from ldm.generate import Generate
        # these two lines prevent a horrible warning message from appearing
        # when the frozen CLIP tokenizer is imported
        import transformers
        transformers.logging.set_verbosity_error()

        self.load_face_restoration()

        try:
            t2i = Generate(
                conf=self.argvopt.config,
                model=self.argvopt.model,
                sampler_name=self.argvopt.sampler_name,
                embedding_path=self.argvopt.embedding_path,
                full_precision=self.argvopt.full_precision,
                precision=self.argvopt.precision,
                gfpgan=self.gfpgan,
                codeformer=self.codeformer,
                esrgan=self.esrgan,
                free_gpu_mem=self.argvopt.free_gpu_mem,
                safety_checker=False,
                max_loaded_models=self.argvopt.max_loaded_models,
            )
        except FileNotFoundError:
            print('** You appear to be missing configs/models.yaml')
            raise
        except (IOError, KeyError) as e:
            print(f'{e}. Aborting.')
            raise

        # make sure the output directory exists
        if not os.path.exists(self.argvopt.outdir):
            os.makedirs(self.argvopt.outdir)

        # gets rid of annoying messages about random seed
        pytorch_logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # preload the model
        t2i.load_model()

        logger.info("Initialization done!")

        return t2i

    def reload_everything(self):
        requested_model = self.t2i.model_cache.models[self.t2i.model_name]['model']
        self.t2i.model_cache.models[self.t2i.model_name]['model'] = self.t2i.model_cache._model_from_cpu(
            requested_model)

    def unload_everything(self):
        self.t2i.model_cache.offload_model(self.t2i.model_name)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print('>> Current VRAM usage: ', '%4.2fG' % (torch.cuda.memory_allocated() / 1e9))

    def load_face_restoration(self):
        try:
            from ldm.invoke.restoration import Restoration
            restoration = Restoration()
            self.gfpgan, self.codeformer = restoration.load_face_restore_models(self.argvopt.gfpgan_model_path)
            self.esrgan = restoration.load_esrgan(self.argvopt.esrgan_bg_tile)
        except (ModuleNotFoundError, ImportError):
            print('>> You may need to install the ESRGAN and/or GFPGAN modules')

    def parse_command(self, command):
        # before splitting, escape single quotes so as not to mess
        # up the parser
        command = command.replace("'", "\\'")
        elements = shlex.split(command)

        if len(elements) == 0:
            raise ValueError

        # rearrange the arguments to mimic how it works in the Dream bot.
        switches = ['']
        switches_started = False

        for el in elements:
            if el[0] == '-' and not switches_started:
                switches_started = True
            if switches_started:
                switches.append(el)
            else:
                switches[0] += el
                switches[0] += ' '
        switches[0] = switches[0][:len(switches[0]) - 1]
        try:
            opt = self.prompt_parser.parse_args(switches)
            # manually set all the argparse parameters the library methods expect
            if opt.width is None:
                opt.width = NORM_DIM
            if opt.height is None:
                opt.height = NORM_DIM
            opt.init_img = None
            opt.skip_normalize = False
            opt.save_original = False
            if opt.cfg_scale <= 1.0:
                opt.cfg_scale = 1.01
        except SystemExit:
            raise ValueError
        if len(opt.prompt) == 0:
            raise ValueError

        if opt.upscale is not None and opt.upscale[0] < 0:
            opt.upscale = None

        if opt.iterations is not None and opt.iterations > 10:
            raise ResourceError("iterations must be equal to or under 10")

        return opt

    async def generator(self, opt, discord_channel):
        logger.info(f"requested generation: {opt}")
        if self.esrgan is None:
            self.load_face_restoration()
        try:
            loop = asyncio.get_running_loop()
            if opt.iterations is None:
                opt.iterations = 1
            if opt.steps is None:
                opt.steps = 16
            if opt.sampler_name is None:
                opt.sampler_name = 'ddim'
            if opt.facetool_strength is None:
                opt.facetool_strength = 0
            iterations = opt.iterations
            opt.iterations = 1
            callback = functools.partial(self.handle_generator_callbacks, opt=opt, discord_channel=discord_channel,
                                         loop=loop)
            for _ in range(iterations):
                await loop.run_in_executor(
                    self.threadpool,
                    functools.partial(
                        self.t2i.prompt2image,
                        image_callback=callback,
                        **vars(opt)
                    )
                )
        except Exception as e:
            logger.error("hit a problem generating", exc_info=e)

    def handle_generator_callbacks(self, image, seed, first_seed=None, use_prefix=None, upscaled=False, loop=None,
                                   opt=None, discord_channel=None):
        if (opt.upscale is not None or opt.facetool_strength > 0) and upscaled is False:
            return
        normalized_prompt = self.normalize_prompt(opt)
        filepath = self.write_jpg(image, seed)
        msg = f"{self.style_text(opt)}`{normalized_prompt}`"
        loop.create_task(
            discord_channel.send(
                msg,
                file=discord.File(filepath),
            )
        )
        logger.info(f"generated: {filepath} {opt}")

    def write_jpg(self, image, seed):
        path = os.path.join(self.argvopt.outdir, f'{time.time():.2f}.{seed}.jpg')
        image.save(path, 'JPEG', quality=95)
        return path

    def style_text(self, opt):
        style = list()
        if opt.sampler_name in SAMPLER_CHOICES:
            style.append(f'Style {SAMPLER_EMOJI[opt.sampler_name]}')
        if opt.cfg_scale > 7.5:
            if opt.cfg_scale > 16:
                style.append('very strict')
            else:
                style.append('strict')
        elif opt.cfg_scale < 7.5:
            if opt.cfg_scale < 4:
                style.append('very lenient')
            else:
                style.append('lenient')
        if opt.sampler_name == 'k_euler_a':
            style.append(f'{opt.steps} steps')
        if len(style) > 0:
            return ', '.join(style) + ': '
        return ''

    def normalize_prompt(self, opt):
        """Normalize the prompt and switches"""
        switches = list()
        switches.append(f'"{opt.prompt}"')
        switches.append(f'-s{opt.steps or self.t2i.steps}')
        if opt.width != NORM_DIM:
            switches.append(f'-W{opt.width or self.t2i.width}')
        if opt.height != NORM_DIM:
            switches.append(f'-H{opt.height or self.t2i.height}')
        switches.append(f'-C{round(opt.cfg_scale or self.t2i.cfg_scale, 2)}')
        switches.append(f'-m{opt.sampler_name or self.t2i.sampler_name}')
        if opt.facetool_strength:
            switches.append(f'-G{opt.facetool_strength}')
        if opt.upscale:
            switches.append(f'-U {" ".join([str(u) for u in opt.upscale])}')
        if opt.seamless:
            switches.append(f'-l')
        if opt.seed:
            switches.append(f'-S{opt.seed}')
        if opt.perlin > 0:
            switches.append(f'--perlin {opt.perlin}')
        return ' '.join(switches)


if __name__ == "__main__":
    bot = DiscordBot()
    bot.run()
