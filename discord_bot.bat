@ECHO OFF
call C:\ProgramData\Miniconda3\Scripts\activate.bat %USERPROFILE%\.conda\envs\sd-ldm
cd %USERPROFILE%\Documents\stable-diffusion-discord
python scripts\discord_bot.py
