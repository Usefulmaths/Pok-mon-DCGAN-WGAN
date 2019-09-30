import requests
from bs4 import BeautifulSoup
from PIL import Image


class Pokemon(object):
    def __init__(self, name, types, image_url):
        self.name = name
        self.types = types
        self.image_url = image_url


def make_lowercase(list):
    new_pokemon_list = []
    for pokemon in pokemon_list:
        new_pokemon_list.append(pokemon.lower())

    return new_pokemon_list


def _download_pokemon(base, pokemon):
    url = base + pokemon + '.png'
    r = requests.get(url, allow_redirects=True)
    open('../data/gen1_pokemon/' + pokemon + '.png', 'wb').write(r.content)


def download_pokemon(base, pokemon_list):
    for pokemon in pokemon_list:
        _download_pokemon(base, pokemon)


def download_pokemon_detailed():
    url = 'https://pokemondb.net/pokedex/national'

    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    infocards = soup.find_all(class_='infocard')

    pokemon_list = []
    for infocard in infocards:
        name = infocard.find(class_='ent-name')['href']
        name = name.replace('/pokedex/', '')

        poke_types = infocard.find_all(class_='itype')
        poke_types = [t.text for t in poke_types]

        image_url = infocard.find(class_='img-sprite')['data-src']

        pokemon = Pokemon(name, poke_types, image_url)
        pokemon_list.append(pokemon)

        r = requests.get(image_url, allow_redirects=True)
        open('../data/gen1_pokemon/all/images/' +
             name + '.png', 'wb').write(r.content)

        png = Image.open('../data/gen1_pokemon/all/images/' +
                         name + '.png').convert('RGBA')
        background = Image.new('RGBA', png.size, (255, 255, 255))

        alpha_composite = Image.alpha_composite(background, png)
        alpha_composite.save('../data/gen1_pokemon/all/images/' +
                             name + '.png', 'PNG', quality=80)
        print("Download: %s" % name)

    return pokemon_list


if __name__ == '__main__':
    '''
    base = "https://img.pokemondb.net/sprites/diamond-pearl/normal/"

    pokemon_list = ['Bulbasaur', 'Ivysaur', 'Venusaur', 'Charmander', 'Charmeleon', 'Charizard', 'Squirtle', 'Wartortle', 'Blastoise', 'Caterpie', 'Metapod', 'Butterfree', 'Weedle', 'Kakuna', 'Beedrill', 'Pidgey', 'Pidgeotto', 'Pidgeot', 'Rattata', 'Raticate', 'Spearow', 'Fearow', 'Ekans', 'Arbok', 'Pikachu', 'Raichu', 'Sandshrew', 'Sandslash', 'Nidoran', 'Nidorina', 'Nidoqueen', 'Nidoran', 'Nidorino', 'Nidoking', 'Clefairy', 'Clefable', 'Vulpix', 'Ninetales', 'Jigglypuff', 'Wigglytuff', 'Zubat', 'Golbat', 'Oddish', 'Gloom', 'Vileplume', 'Paras', 'Parasect', 'Venonat', 'Venomoth', 'Diglett', 'Dugtrio', 'Meowth', 'Persian', 'Psyduck', 'Golduck', 'Mankey', 'Primeape', 'Growlithe', 'Arcanine', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Abra', 'Kadabra', 'Alakazam', 'Machop', 'Machoke', 'Machamp', 'Bellsprout', 'Weepinbell', 'Victreebel', 'Tentacool', 'Tentacruel',
                    'Geodude', 'Graveler', 'Golem', 'Ponyta', 'Rapidash', 'Slowpoke', 'Slowbro', 'Magnemite', 'Magneton', 'Farfetch', 'Doduo', 'Dodrio', 'Seel', 'Dewgong', 'Grimer', 'Muk', 'Shellder', 'Cloyster', 'Gastly', 'Haunter', 'Gengar', 'Onix', 'Drowzee', 'Hypno', 'Krabby', 'Kingler', 'Voltorb', 'Electrode', 'Exeggcute', 'Exeggutor', 'Cubone', 'Marowak', 'Hitmonlee', 'Hitmonchan', 'Lickitung', 'Koffing', 'Weezing', 'Rhyhorn', 'Rhydon', 'Chansey', 'Tangela', 'Kangaskhan', 'Horsea', 'Seadra', 'Goldeen', 'Seaking', 'Staryu', 'Starmie', 'Mr', 'Scyther', 'Jynx', 'Electabuzz', 'Magmar', 'Pinsir', 'Tauros', 'Magikarp', 'Gyarados', 'Lapras', 'Ditto', 'Eevee', 'Vaporeon', 'Jolteon', 'Flareon', 'Porygon', 'Omanyte', 'Omastar', 'Kabuto', 'Kabutops', 'Aerodactyl', 'Snorlax', 'Articuno', 'Zapdos', 'Moltres', 'Dratini', 'Dragonair', 'Dragonite', 'Mewtwo', 'Mew']

    pokemon_list = make_lowercase(pokemon_list)

    download_pokemon(base, pokemon_list)
    '''
    download_pokemon_detailed()
