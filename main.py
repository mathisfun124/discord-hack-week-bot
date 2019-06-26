import discord
from config import token
from discord.ext import commands
from sample import sample, sample_args

description = 'A small bot that does strange things.'

bot = commands.Bot(command_prefix='!', description=description)


@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('----------')


#  Note that the command is very slow
@bot.command()
async def sao(ctx):
    await ctx.send("Loading text...")
    text = await sample(sample_args)
    await ctx.send(text)

bot.remove_command('help')


@bot.command()
async def help(ctx):
    embed = discord.Embed(title='Help')
    embed.add_field(name='sao', value='Creates a small snippet of text from SAO using a neural network model.')
    await ctx.send(embed=embed)


bot.run(token)
