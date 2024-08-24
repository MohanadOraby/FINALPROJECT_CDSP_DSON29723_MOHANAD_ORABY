import streamlit as st
import pandas as pd
import pickle

# Load the trained model and preprocessing pipeline
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('final_columns.pkl', 'rb') as columns_file:
    original_columns = pickle.load(columns_file)

# Preprocessing function for aligning user input with training columns
def preprocess_input(data, original_columns):
    """
    Preprocesses the user input by applying pd.get_dummies and aligning it with the original training columns.
    Parameters:
    - data: A DataFrame containing the user's input (one row).
    - original_columns: The list of columns used during training (including dummy columns).
    
    Returns:
    - preprocessed_data: A DataFrame with the same columns as the original training data.
    """
    # Apply pd.get_dummies to the user input (only for categorical columns)
    data_dummies = pd.get_dummies(data, columns=['sector', 'age_of_property', 'locality', 'nearest_tm_station'])
    
    # Align the dummy columns with the original columns used during training
    # If some columns are missing, fill them with 0
    preprocessed_data = data_dummies.reindex(columns=original_columns, fill_value=0)
    
    return preprocessed_data

# Image of Bogotá
st.image("Bogota_image.jpeg", width=700)

# Streamlit app interface
st.title("Bogotá Property Price Prediction")
st.subheader("FINALPROJECT_CDSP_DSON29723_MOHANAD_ORABY")

# Collecting input from the user
area = st.number_input('Area (in square meters)', min_value=10, max_value=1000, step=1)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, step=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, step=1)
administration_fee = st.number_input('Administration Fee (in $COP)', min_value=0, step=1000)
parking_spaces = st.selectbox('Number of Parking Spaces', [0,1,2,3,4])
socioeconomic_level = st.selectbox('Socioeconomic_level', [1,2,3,4,5,6])
age_of_property = st.selectbox('Age of Property', ['ENTRE 10 Y 20 ANOS', 'MAS DE 20 ANOS', 'ENTRE 0 Y 5 ANOS',
       'ENTRE 5 Y 10 ANOS', 'REMODELADO'])

longitude = st.number_input('Longitude',format="%.3f")
latitude = st.number_input('Latitude',format="%.3f")

jacuzzi = st.selectbox('Has Jacuzzi?', [0, 1])
floor = st.number_input('Floor', min_value=1, step=1)
closets = st.number_input('Number of Closets', min_value=0, step=1)
fireplace = st.selectbox('Has Fireplace?', [0, 1])
pets_allowed = st.selectbox('Allows Pets?', [0, 1])
gym = st.selectbox('Has Gym?', [0, 1])
elevator = st.selectbox('Has Elevator?', [0, 1])
gated_community = st.selectbox('Gated Community?', [0, 1])

locality = st.selectbox('Locality', ['ANTONIO NARINO', 'BARRIOS UNIDOS', 'BOSA', 'CANDELARIA',
			'CHAPINERO', 'CIUDAD BOLIVAR', 'ENGATIVA', 'FONTIBON', 'KENNEDY', 'LOS MARTIRES',
			'PUENTE ARANDA', 'RAFAEL URIBE URIBE', 'SAN CRISTOBAL', 'SANTA FE', 'SUBA',
			'TEUSAQUILLO', 'TUNJUELITO', 'USAQUEN', 'USME'])

sector = st.selectbox('Sector', ['170 Y ALREDORES', 'ALTOS DE SUBA Y CERROS DE SAN JORGE', 'AMERICAS',
 			'ANTONIO NARINO', 'APOGEO', 'ARBORIZADORA', 'BAVARIA', 'BOLIVIA',
 			'BOSA CENTRAL', 'BOSA OCCIDENTAL', 'BOSA SOACHA', 'BOYACA REAL', 'BRITALIA',
 			'CALANDAIMA', 'CASA BLANCA SUBA', 'CASTILLA', 'CASTILLA MARSELLA', 'CEDRITOS',
 			'CENTRO INTERNACIONAL', 'CENTRO NARINO', 'CENTRO Y ZONA COLONIAL', 'CERROS DE SUBA',
 			'CHAPINERO', 'CHAPINERO ALTO', 'CHICO', 'CHICO LAGO', 'CIUDAD BOLIVAR',
 			'CIUDAD SALITRE OCCIDENTAL', 'CIUDAD SALITRE ORIENTAL', 'CIUDAD USME', 'COLINA Y ALREDEDORES',
 			'CORTIJO AUTOPISTA MEDELLIN', 'COUNTRY', 'COUNTRY CLUB', 'DANUBIO', 'DOCE DE OCTUBRE',
 			'EL MINUTO DE DIOS', 'EL PORVENIR', 'EL PRADO', 'EL REFUGIO', 'EL RINCON', 'ENGATIVA',
 			'FONTIBON', 'FONTIBON SAN PABLO', 'FONTIBON TINTAL', 'GALERIAS', 'GARCES NAVAS', 'GRAN BRITALIA',
 			'GRANJAS DE TECHO', 'GUAYMARAL', 'ISMAEL PERDOMO', 'KENNEDY', 'LA ALHAMBRA', 'LA ESMERALDA',
 			'LA FLORESTA', 'LA SOLEDAD', 'LA URIBE', 'LAS MARGARITAS', 'LOS ALCAZARES', 'LOS CEDROS', 'MARRUECOS',
 			'METROPOLIS', 'MODELIA', 'NICOLAS DE FEDERMAN', 'NIZA', 'NIZA ALHAMBRA', 'NORMANDIA', 'ORQUIDEAS',
 			'OTROS', 'PARDO RUBIO', 'PUENTE ARANDA', 'QUINTA PAREDES', 'QUIROGA', 'RAFAEL URIBE URIBE TUNJUELITO',
 			'RESTREPO', 'SAGRADO CORAZON', 'SALITRE MODELIA', 'SAN JOSE DE BAVARIA', 'SAN RAFAEL', 'SANTA BARBARA',
 			'SANTA CECILIA', 'SOSIEGO', 'SUBA', 'TIBABUYES', 'TIMIZA', 'TIMIZA LA ALQUERIA', 'TINTAL NORTE',
 			'TINTAL SUR', 'TOBERIN', 'USAQUEN', 'VENECIA', 'VERBENAL', 'ZONA FRANCA'])

nearest_tm_station = st.selectbox('Nearest TM Station', ['21 Ángeles', '7 de Agosto', 'AV. 1 Mayo', 'AV. 39',
 			'AV. 68', 'AV. Américas - AV. Boyacá', 'AV. Boyacá', 'AV. Cali', 'AV. Chile',
 			'AV. El Dorado', 'AV. Jiménez - CL 13', 'AV. Jiménez - Caracas', 'AV. Rojas',
 			'Alcalá', 'Alquería', 'Banderas', 'Biblioteca', 'Biblioteca Tintal', 'Bicentenario',
 			'Bosa', 'CAD', 'CAN', 'Calle 100 - Marketmedios', 'Calle 106', 'Calle 127', 'Calle 142',
 			'Calle 146', 'Calle 161', 'Calle 187', 'Calle 19', 'Calle 22', 'Calle 26', 'Calle 34', 'Calle 40 S',
 			'Calle 45 - American School Way', 'Calle 57', 'Calle 63', 'Calle 72', 'Calle 76 - San Felipe', 'Calle 85',
 			'Campín - UAN', 'Carrera 43 - Comapan', 'Carrera 47', 'Carrera 53', 'Carrera 90',
 			'Centro Comercial Paseo Villa del Río - Madelena', 'Centro Memoria', 'Ciudad Jardín - UAN',
 			'Ciudad Universitaria - Lotería de Bogotá', 'Comuneros', 'Concejo de Bogotá', 'Country Sur',
 			'De La Sabana', 'Distrito Grafiti', 'El Tiempo - Maloka', 'Ferias', 'Flores', 'Fucha',
 			'General Santander', 'Gobernación', 'Granja - Carrera 77', 'Gratamira', 'Guatoque - Veraguas',
 			'Humedal Córdoba', 'Héroes - Gel´Hada', 'La Campiña', 'La Despensa', 'Las Aguas', 'Las Nieves',
 			'Leon XIII', 'Mandalay', 'Marly', 'Marsella', 'Mazurén', 'Minuto de Dios', 'Modelia', 'Molinos',
 			'Movistar Arena', 'Museo Nacional', 'Museo del Oro', 'NQS - Calle 30 S', 'NQS - Calle 38A S',
 			'Niza - Calle 127', 'Normandía', 'Olaya', 'Paloquemao', 'Patio Bonito', 'Pepe Sierra',
 			'Perdomo', 'Policarpa', 'Polo', 'Portal 20 de Julio', 'Portal 80', 'Portal Américas',
 			'Portal El Dorado', 'Portal Norte', 'Portal Suba', 'Portal Sur - JFK Coop. Financiera',
 			'Portal Tunal', 'Portal Usme', 'Pradera', 'Prado', 'Puente Aranda', 'Puentelargo',
 			'Quinta Paredes', 'Quirigua', 'Quiroga', 'Recinto Ferial', 'Restrepo', 'Ricaurte - CL 13',
 			'Ricaurte - NQS', 'SENA', 'Salitre - El Greco', 'San Diego', 'San Fason Carrera 22', 'Santa Isabel',
 			'Santa Lucía', 'Sevillana', 'Suba - AV. Boyacá', 'Suba - Calle 100', 'Suba - Calle 116',
 			'Suba - Calle 95', 'Suba - TV. 91', 'Terminal', 'Terreros - Hospital C.V', 'Toberín - Foundever',
 			'Transversal 86', 'Tygua - San José', 'U. Nacional', 'Universidades', 'Virrey', 'Zona Industrial'])


# Dictionary for the input
user_input = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'administration_fee': administration_fee,
    'parking_spaces': parking_spaces,
    'sector': sector,
    'socioeconomic_level' : socioeconomic_level,
    'age_of_property': age_of_property,
    'locality': locality,
    'nearest_tm_station': nearest_tm_station,
    'longitude': longitude,
    'latitude': latitude,
    'jacuzzi': jacuzzi,
    'floor': floor,
    'closets': closets,
    'fireplace': fireplace,
    'pets_allowed': pets_allowed,
    'gym': gym,
    'elevator': elevator,
    'gated_community': gated_community
}

# Converting user input to a DataFrame
user_df = pd.DataFrame([user_input])

# Button called 'Predict' and shows prediction based on user parameters
if st.button('Predict'):
    try:
        preprocessed_input = preprocess_input(user_df, original_columns)
        prediction = model.predict(preprocessed_input)
        
        # Calculate 10% range
        lower_bound = prediction[0] * 0.9
        upper_bound = prediction[0] * 1.1

        # Display the results
        st.write(f"Predicted Price COP {prediction[0]:,.2f}")
        st.write(f"Predicted Price Range (-10% to +10%): COP {lower_bound:,.2f} - COP {upper_bound:,.2f}")

    except Exception as e:
        st.write(f"Error in making prediction: {e}")
