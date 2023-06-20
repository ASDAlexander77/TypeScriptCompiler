function save(_response: IRootResponse<{ field1: string }>): void {};

function exportCommand<TResponse>(functionToCall: IExportCallback<TResponse>): void {};

interface IExportCallback<TResponse> {
	(response: IRootResponse<TResponse>): void;
}

type IRootResponse<TResponse> =
	TResponse extends IRecord ? IRecordResponse<TResponse> : IResponse<TResponse>;

interface IRecord {
	readonly Id: string;
}

declare type IRecordResponse<T extends IRecord> = IResponse<T> & {
	sendRecord(): void;
};

declare type IResponse<T> = {
	sendValue(name: keyof GetAllPropertiesOfType<T, string>): void;
};

declare type GetPropertyNamesOfType<T, RestrictToType> = {
	[PropertyName in Extract<keyof T, string>]: T[PropertyName] extends RestrictToType ? PropertyName : never
}[Extract<keyof T, string>];

declare type GetAllPropertiesOfType<T, RestrictToType> = Pick<
	T,
	GetPropertyNamesOfType<Required<T>, RestrictToType>
>;

function main() {
    print("done.");
}